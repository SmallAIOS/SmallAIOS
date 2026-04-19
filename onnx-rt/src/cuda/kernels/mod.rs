// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NVRTC-backed kernel compilation and launch infrastructure.
//!
//! Provides `compile_kernel` (JIT CUDA source -> PTX -> CUmodule) and
//! `launch_kernel` (safe wrapper around `cuLaunchKernel`). Every compiled
//! kernel is represented as a [`Kernel`] value that owns a `CUmodule` and
//! caches the resolved `CUfunction` handle for the lifetime of the struct.
//!
//! This module requires an active CUDA driver-API context. Because NVRTC
//! and the driver API do not share context state with the CUDA runtime
//! (`cudart`) automatically, [`lazy_context_init`] bootstraps the primary
//! context on device 0 once per process via `std::sync::Once`.

extern crate alloc;

use alloc::ffi::CString;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use std::sync::Once;

use super::ffi;
use super::CudaError;

pub mod attention;
pub mod elementwise;
pub mod gather;
pub mod rms_norm;
pub mod rotary;

/// Default NVRTC compilation options used when a caller passes an empty
/// options slice. Targets Blackwell GB10 (sm_121).
///
/// Callers that need a different architecture can pass their own options
/// via [`compile_kernel`]'s `options` parameter (it overrides this default
/// when non-empty).
pub const DEFAULT_NVRTC_OPTIONS: &[&str] = &[
    "--gpu-architecture=sm_121",
    "-I/usr/local/cuda/include",
    "--std=c++17",
];

/// A compiled CUDA kernel, ready to launch.
///
/// Owns a driver-API `CUmodule` and caches a resolved `CUfunction` handle.
/// Dropping the value unloads the module via `cuModuleUnload`.
pub struct Kernel {
    name: String,
    module: ffi::CUmodule,
    function: ffi::CUfunction,
}

// Safety: CUDA driver-API modules and functions are safe to move across
// threads as long as the containing context is current on the calling
// thread. SmallAIOS serializes kernel launches on the default stream.
unsafe impl Send for Kernel {}

impl Kernel {
    /// Human-readable kernel name (also the NVRTC symbol name).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw `CUfunction` handle (for advanced direct launches).
    pub fn function(&self) -> ffi::CUfunction {
        self.function
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // Safety: module was created by cuModuleLoadData and is unloaded
            // exactly once here.
            unsafe {
                let _ = ffi::cuModuleUnload(self.module);
            }
        }
    }
}

// ── Context bootstrap ───────────────────────────────────────────────

static CTX_INIT: Once = Once::new();
static mut CTX_INIT_RESULT: Result<(), CudaError> = Ok(());

/// Ensure the CUDA driver-API primary context on device 0 is retained and
/// set as current on the calling thread. Runs at most once per process.
///
/// This is required because NVRTC and the driver API cannot use a context
/// that was implicitly created by the runtime API (`cudart`) on a different
/// thread — safer to own a driver-API primary context explicitly.
pub(crate) fn lazy_context_init() -> Result<(), CudaError> {
    fn do_init() -> Result<(), CudaError> {
        // Safety: FFI calls; we check every return code.
        unsafe {
            let rc = ffi::cuInit(0);
            if rc != ffi::CU_SUCCESS {
                return Err(CudaError::RuntimeError {
                    op: "cuInit",
                    code: rc,
                });
            }
            let mut dev: ffi::CUdevice = 0;
            let rc = ffi::cuDeviceGet(&mut dev, 0);
            if rc != ffi::CU_SUCCESS {
                return Err(CudaError::RuntimeError {
                    op: "cuDeviceGet",
                    code: rc,
                });
            }
            let mut ctx: ffi::CUcontext = core::ptr::null_mut();
            let rc = ffi::cuDevicePrimaryCtxRetain(&mut ctx, dev);
            if rc != ffi::CU_SUCCESS {
                return Err(CudaError::RuntimeError {
                    op: "cuDevicePrimaryCtxRetain",
                    code: rc,
                });
            }
            let rc = ffi::cuCtxSetCurrent(ctx);
            if rc != ffi::CU_SUCCESS {
                return Err(CudaError::RuntimeError {
                    op: "cuCtxSetCurrent",
                    code: rc,
                });
            }
        }
        Ok(())
    }

    CTX_INIT.call_once(|| {
        let result = do_init();
        // Safety: CTX_INIT_RESULT is written exactly once inside call_once.
        unsafe {
            CTX_INIT_RESULT = result;
        }
    });
    // Safety: after call_once completes, the value is fully published.
    unsafe {
        match &*core::ptr::addr_of!(CTX_INIT_RESULT) {
            Ok(()) => Ok(()),
            Err(e) => Err(clone_cuda_error(e)),
        }
    }
}

fn clone_cuda_error(e: &CudaError) -> CudaError {
    match e {
        CudaError::NoDevice => CudaError::NoDevice,
        CudaError::VersionMismatch {
            expected_major,
            actual_major,
        } => CudaError::VersionMismatch {
            expected_major: *expected_major,
            actual_major: *actual_major,
        },
        CudaError::AllocFailed { size, code } => CudaError::AllocFailed {
            size: *size,
            code: *code,
        },
        CudaError::CopyFailed { msg, code } => CudaError::CopyFailed { msg, code: *code },
        CudaError::BlasError { op, code } => CudaError::BlasError { op, code: *code },
        CudaError::DnnError { op, code } => CudaError::DnnError { op, code: *code },
        CudaError::RuntimeError { op, code } => CudaError::RuntimeError { op, code: *code },
        CudaError::KernelCompileFailed { name, log } => CudaError::KernelCompileFailed {
            name: name.clone(),
            log: log.clone(),
        },
        CudaError::KernelLoadFailed { name, cuda_result } => CudaError::KernelLoadFailed {
            name: name.clone(),
            cuda_result: *cuda_result,
        },
        CudaError::KernelLaunchFailed { name, cuda_result } => CudaError::KernelLaunchFailed {
            name: name.clone(),
            cuda_result: *cuda_result,
        },
    }
}

// ── Compile ────────────────────────────────────────────────────────

/// Compile CUDA C++ source to PTX via NVRTC, load it as a driver-API
/// module, and resolve the function symbol `name`.
///
/// `name` doubles as the NVRTC program name **and** the kernel entry-point
/// symbol. Kernels must be declared `extern "C" __global__ void <name>(...)`
/// to avoid C++ name mangling.
///
/// When `options` is empty, [`DEFAULT_NVRTC_OPTIONS`] is used (targets
/// `sm_121` for Blackwell GB10). Pass a non-empty slice to override.
///
/// Errors:
/// - [`CudaError::KernelCompileFailed`] if NVRTC rejects the source; the
///   `log` field contains the NVRTC build log.
/// - [`CudaError::KernelLoadFailed`] if `cuModuleLoadData` or
///   `cuModuleGetFunction` fails after a successful compile.
/// - [`CudaError::RuntimeError`] if lazy context initialization fails.
pub fn compile_kernel(name: &str, source: &str, options: &[&str]) -> Result<Kernel, CudaError> {
    lazy_context_init()?;

    let source_c = CString::new(source).map_err(|_| CudaError::KernelCompileFailed {
        name: name.to_string(),
        log: String::from("kernel source contained interior NUL byte"),
    })?;
    let name_c = CString::new(name).map_err(|_| CudaError::KernelCompileFailed {
        name: name.to_string(),
        log: String::from("kernel name contained interior NUL byte"),
    })?;

    // NVRTC takes an array of C-string pointers. Build the CStrings first so
    // they live for the duration of the compile call.
    let effective_options: &[&str] = if options.is_empty() {
        DEFAULT_NVRTC_OPTIONS
    } else {
        options
    };
    let mut option_cstrings: Vec<CString> = Vec::with_capacity(effective_options.len());
    for opt in effective_options {
        let c = CString::new(*opt).map_err(|_| CudaError::KernelCompileFailed {
            name: name.to_string(),
            log: format!("NVRTC option {opt:?} contained interior NUL byte"),
        })?;
        option_cstrings.push(c);
    }
    let option_ptrs: Vec<*const core::ffi::c_char> =
        option_cstrings.iter().map(|c| c.as_ptr()).collect();

    // Create program.
    let mut prog: ffi::nvrtcProgram = core::ptr::null_mut();
    let rc = unsafe {
        ffi::nvrtcCreateProgram(
            &mut prog,
            source_c.as_ptr(),
            name_c.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        )
    };
    if rc != ffi::NVRTC_SUCCESS {
        return Err(CudaError::KernelCompileFailed {
            name: name.to_string(),
            log: nvrtc_error_string(rc),
        });
    }

    // Compile.
    let rc =
        unsafe { ffi::nvrtcCompileProgram(prog, option_ptrs.len() as i32, option_ptrs.as_ptr()) };
    if rc != ffi::NVRTC_SUCCESS {
        let log = fetch_nvrtc_log(prog).unwrap_or_else(|| nvrtc_error_string(rc));
        unsafe {
            let _ = ffi::nvrtcDestroyProgram(&mut prog);
        }
        return Err(CudaError::KernelCompileFailed {
            name: name.to_string(),
            log,
        });
    }

    // Retrieve PTX.
    let mut ptx_size: usize = 0;
    let rc = unsafe { ffi::nvrtcGetPTXSize(prog, &mut ptx_size) };
    if rc != ffi::NVRTC_SUCCESS || ptx_size == 0 {
        let log = fetch_nvrtc_log(prog).unwrap_or_else(|| nvrtc_error_string(rc));
        unsafe {
            let _ = ffi::nvrtcDestroyProgram(&mut prog);
        }
        return Err(CudaError::KernelCompileFailed {
            name: name.to_string(),
            log,
        });
    }
    let mut ptx: Vec<u8> = vec![0u8; ptx_size];
    let rc = unsafe { ffi::nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut core::ffi::c_char) };
    if rc != ffi::NVRTC_SUCCESS {
        let log = fetch_nvrtc_log(prog).unwrap_or_else(|| nvrtc_error_string(rc));
        unsafe {
            let _ = ffi::nvrtcDestroyProgram(&mut prog);
        }
        return Err(CudaError::KernelCompileFailed {
            name: name.to_string(),
            log,
        });
    }

    // NVRTC program no longer needed; the PTX buffer is self-contained.
    unsafe {
        let _ = ffi::nvrtcDestroyProgram(&mut prog);
    }

    // Load PTX image into a driver-API module.
    let mut module: ffi::CUmodule = core::ptr::null_mut();
    let rc =
        unsafe { ffi::cuModuleLoadData(&mut module, ptx.as_ptr() as *const core::ffi::c_void) };
    if rc != ffi::CU_SUCCESS {
        return Err(CudaError::KernelLoadFailed {
            name: name.to_string(),
            cuda_result: rc,
        });
    }

    // Resolve the function symbol.
    let mut function: ffi::CUfunction = core::ptr::null_mut();
    let rc = unsafe { ffi::cuModuleGetFunction(&mut function, module, name_c.as_ptr()) };
    if rc != ffi::CU_SUCCESS {
        unsafe {
            let _ = ffi::cuModuleUnload(module);
        }
        return Err(CudaError::KernelLoadFailed {
            name: name.to_string(),
            cuda_result: rc,
        });
    }

    Ok(Kernel {
        name: name.to_string(),
        module,
        function,
    })
}

fn fetch_nvrtc_log(prog: ffi::nvrtcProgram) -> Option<String> {
    if prog.is_null() {
        return None;
    }
    let mut log_size: usize = 0;
    let rc = unsafe { ffi::nvrtcGetProgramLogSize(prog, &mut log_size) };
    if rc != ffi::NVRTC_SUCCESS || log_size <= 1 {
        return None;
    }
    let mut buf: Vec<u8> = vec![0u8; log_size];
    let rc = unsafe { ffi::nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut core::ffi::c_char) };
    if rc != ffi::NVRTC_SUCCESS {
        return None;
    }
    // Strip trailing NUL.
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn nvrtc_error_string(rc: ffi::nvrtcResult) -> String {
    let ptr = unsafe { ffi::nvrtcGetErrorString(rc) };
    if ptr.is_null() {
        return format!("nvrtc error code {rc}");
    }
    // Safety: NVRTC returns a static NUL-terminated string.
    unsafe {
        let cstr = core::ffi::CStr::from_ptr(ptr);
        cstr.to_string_lossy().into_owned()
    }
}

// ── Launch ─────────────────────────────────────────────────────────

/// Launch `kernel` on the default stream.
///
/// `args` must point at packed kernel-argument slots exactly as required
/// by `cuLaunchKernel`: a slice of `*mut c_void` where each element is the
/// address of a local variable holding the argument value (including device
/// pointers — store the pointer in a local, then take `&mut` of it).
///
/// This call is asynchronous; use [`synchronize`] to block until the kernel
/// finishes if you need the results on the host.
pub fn launch_kernel(
    kernel: &Kernel,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    args: &[*mut core::ffi::c_void],
    shared_bytes: u32,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::cuLaunchKernel(
            kernel.function,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_bytes,
            core::ptr::null_mut(),
            args.as_ptr() as *mut *mut core::ffi::c_void,
            core::ptr::null_mut(),
        )
    };
    if rc != ffi::CU_SUCCESS {
        return Err(CudaError::KernelLaunchFailed {
            name: kernel.name.clone(),
            cuda_result: rc,
        });
    }
    Ok(())
}

/// Block the calling thread until all previously launched kernels on the
/// current driver-API context have completed. Uses `cuCtxSynchronize`.
pub fn synchronize() -> Result<(), CudaError> {
    let rc = unsafe { ffi::cuCtxSynchronize() };
    if rc != ffi::CU_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cuCtxSynchronize",
            code: rc,
        });
    }
    Ok(())
}
