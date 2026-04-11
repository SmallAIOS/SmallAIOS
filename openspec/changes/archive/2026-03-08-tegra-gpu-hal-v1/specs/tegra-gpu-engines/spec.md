## Tegra GPU Engine Initialization

### Overview

After firmware is loaded, the GPU's GR (graphics/compute) engine, FIFO channels, and GMMU (GPU Memory Management Unit) must be initialized before compute kernels can be dispatched. The GM20B has 1 GPC, 2 TPCs, 4 SMs (128 CUDA cores total).

### GR Engine Init

1. Reset GR engine via `PMC_ENABLE` (toggle GR reset bit)
2. Wait for FECS to report idle via `FECS_CTXSW_STATUS`
3. Query hardware topology:
   - Read `NV_PGRAPH_GPC_COUNT` (1 for GM20B)
   - Read `NV_PGRAPH_TPC_PER_GPC` (2 for GM20B)
   - Read `NV_PGRAPH_SM_PER_TPC` (2 for GM20B)
4. Configure ZCULL: allocate and program ZCULL RAM region
5. Configure attribute circular buffer (attrib CB): allocate per-TPC buffers
6. Generate golden context image:
   - Allocate context buffer (FECS method `ALLOCATE_GR_CONTEXT`)
   - Set context buffer address via FECS mailbox
   - Trigger FECS method `SET_GOLDEN_IMAGE`
   - Save golden context for future channel context init

### FIFO Channel Init

The GM20B has 1 PBDMA (Push Buffer DMA) engine. We allocate a single compute channel:

1. Enable FIFO engine via `NV_PFIFO_ENABLE`
2. Allocate channel instance block (IB) in DRAM — 4 KB aligned
3. Configure PBDMA:
   - Set instance block address: `NV_PPBDMA_CHAN_INST`
   - Set push buffer base and size: `NV_PPBDMA_PB_BASE`, `NV_PPBDMA_PB_SIZE`
   - Set GP (GPFIFO) entry base and count: `NV_PPBDMA_GP_BASE`, `NV_PPBDMA_GP_SIZE`
4. Bind channel to GR engine via `NV_PCCSR_CHANNEL_BIND`
5. Enable channel: set ENABLE bit in channel control register

### GMMU Page Table Setup

The GM20B GMMU uses a two-level page table:

- **PDB (Page Directory Base):** 4 KB, 1024 entries, each pointing to a small page table
- **Small page table:** 4 KB, 1024 entries, each mapping a 4 KB page
- **Address space:** 40-bit GPU virtual (1 TB), but we only map the DRAM region

Initial setup (identity mapping):
1. Allocate PDB (physically contiguous, 4 KB aligned)
2. For each 4 MB region of the GPU-accessible DRAM range:
   - Allocate a small page table
   - Fill entries with identity-mapped physical addresses
   - Write PDB entry pointing to the small page table
3. Program `NV_PGRAPH_PDB_BASE` and `NV_PFIFO_PDB_BASE` with PDB physical address
4. Invalidate TLB via `NV_PFB_PRI_MMU_INVALIDATE`

### Interface

```rust
pub struct GrEngine {
    gpc_count: u32,
    tpc_per_gpc: u32,
    sm_count: u32,
    golden_ctx: Option<u64>,  // Physical address of golden context
    initialized: bool,
}

pub struct FifoChannel {
    channel_id: u32,
    instance_block: u64,  // Physical address
    pushbuffer: u64,      // Physical address
    gp_entries: u64,      // Physical address
    bound: bool,
}

pub struct GmmuPageTable {
    pdb_base: u64,        // Physical address of PDB
    mapped_size: u64,     // Total mapped bytes
}

impl GrEngine {
    pub fn init(bar0_base: usize) -> Result<Self, GpuError>;
    pub fn topology(&self) -> (u32, u32, u32); // (GPC, TPC, SM)
}

impl FifoChannel {
    pub fn allocate(bar0_base: usize) -> Result<Self, GpuError>;
    pub fn bind_to_gr(&mut self) -> Result<(), GpuError>;
    pub fn submit_work(&mut self, gpfifo_entry: u64) -> Result<(), GpuError>;
}

impl GmmuPageTable {
    pub fn new_identity(dram_base: u64, dram_size: u64) -> Result<Self, GpuError>;
    pub fn invalidate_tlb(bar0_base: usize) -> Result<(), GpuError>;
}
```

### Verification

- Unit tests for GR topology parsing (1 GPC, 2 TPC, 4 SM)
- Unit tests for FIFO instance block layout and PBDMA register calculations
- Unit tests for GMMU PDB/PTE entry construction (address bits, valid bits, page size)
- Unit tests for identity mapping correctness (GPU VA == physical addr)
- Unit tests for TLB invalidation sequence
- Unit tests for GPFIFO entry format (address, length, opcode fields)
