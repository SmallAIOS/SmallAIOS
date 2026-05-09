// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Reusable code paths for `cargo xtask scaffold-config-field`.
//!
//! Spec context:
//! `openspec/changes/management-login-v1/specs/mgmt-config-surface-trait/spec.md`
//! Q3 — "developer-ergonomics scaffolder for new `Config` fields".
//!
//! The scaffolder prints the boilerplate diff a developer needs to
//! copy into the relevant files. Real auto-application is a future
//! polish; for v1 this is a copy-paste helper.

use std::fmt;

/// Supported field types — matches `mgmt::config::registry::FieldType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// `Value::Bool`
    Bool,
    /// `Value::U32`
    U32,
    /// `Value::U64`
    U64,
    /// `Value::String`
    String,
    /// `Value::PasswordClasses`
    PasswordClasses,
}

impl FieldType {
    /// Parse a CLI argument like `--type u32` into a [`FieldType`].
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "bool" => Ok(Self::Bool),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "string" => Ok(Self::String),
            "password_classes" => Ok(Self::PasswordClasses),
            other => Err(format!(
                "unknown --type '{other}': expected bool|u32|u64|string|password_classes"
            )),
        }
    }

    /// String form used in the registry's `FieldType::*` enum variant.
    pub fn enum_variant(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::String => "String",
            Self::PasswordClasses => "PasswordClasses",
        }
    }

    /// Variant tag used in `Value::*` (matches the runtime tag).
    pub fn value_variant(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::String => "String",
            Self::PasswordClasses => "PasswordClasses",
        }
    }

    /// Rust type used in the namespace struct field.
    pub fn rust_type(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::String => "alloc::string::String",
            Self::PasswordClasses => "PasswordClassSet",
        }
    }
}

/// Reload semantics — matches `mgmt::surface::ReloadKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadKind {
    /// `ReloadKind::Live`
    Live,
    /// `ReloadKind::Boot`
    Boot,
}

impl ReloadKind {
    /// Parse `--reload live|boot`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "live" => Ok(Self::Live),
            "boot" => Ok(Self::Boot),
            other => Err(format!("unknown --reload '{other}': expected live|boot")),
        }
    }

    /// Variant name used in the registry.
    pub fn enum_variant(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Boot => "Boot",
        }
    }
}

/// All inputs required to scaffold a new field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaffoldArgs {
    /// Dotted path, e.g. `idle.warning_minutes`.
    pub path: String,
    /// Field's typed [`FieldType`].
    pub kind: FieldType,
    /// Reload semantics.
    pub reload: ReloadKind,
}

impl ScaffoldArgs {
    /// Top-level namespace — `path.split('.').next()`.
    pub fn namespace(&self) -> &str {
        self.path.split('.').next().unwrap_or(&self.path)
    }

    /// Trailing field name (e.g. `warning_minutes`).
    pub fn field_name(&self) -> &str {
        self.path.rsplit('.').next().unwrap_or(&self.path)
    }
}

impl fmt::Display for ScaffoldArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ScaffoldArgs {{ path: {}, kind: {:?}, reload: {:?} }}",
            self.path, self.kind, self.reload
        )
    }
}

/// Render the human-facing diff a developer copy-pastes into the
/// codebase. Output is a single `String` so callers can `println!` it
/// or write it to a file under test.
pub fn render_diff(args: &ScaffoldArgs) -> String {
    let ns = args.namespace();
    let name = args.field_name();
    let mut out = String::new();

    out.push_str("# scaffold-config-field — paste the snippets below into\n");
    out.push_str("# the named files. The build will then run the\n");
    out.push_str("# universal-exposure walker; if you missed a step, the\n");
    out.push_str("# walker fails the build with a precise diagnostic.\n");
    out.push('\n');

    // 1. Namespace struct field
    out.push_str("## 1. mgmt/src/config.rs — append a field to the namespace struct\n\n");
    out.push_str(&format!(
        "// Inside `{ns}Config`:\n    pub {name}: {ty},\n",
        ty = args.kind.rust_type()
    ));
    out.push_str("\n// And populate the v1 default:\n");
    out.push_str(&format!("    {name}: <CHOOSE A SAFE DEFAULT>,\n",));
    out.push('\n');

    // 2. read_path arm
    out.push_str("## 2. mgmt/src/config.rs — extend `Config::read_path`\n\n");
    out.push_str(&format!(
        "    \"{path}\" => Ok(Value::{variant}(self.{ns}.{name}.clone())),\n",
        path = args.path,
        variant = args.kind.value_variant(),
        ns = ns,
        name = name
    ));
    out.push('\n');

    // 3. write_path arm
    out.push_str("## 3. mgmt/src/config.rs — extend `Config::write_path`\n\n");
    out.push_str(&format!(
        "    \"{path}\" => self.{ns}.{name} = value.as_<rust>()?,\n",
        path = args.path,
        ns = ns,
        name = name
    ));
    out.push('\n');

    // 4. Validator
    out.push_str("## 4. mgmt/src/validate.rs — extend `validate_field`\n\n");
    out.push_str(&format!(
        "    \"{path}\" => {{ /* per-field rule */ Ok(()) }}\n",
        path = args.path
    ));
    out.push('\n');

    // 5. Registry row
    out.push_str("## 5. mgmt/src/config.rs — append to `registry::CONFIG_FIELDS`\n\n");
    out.push_str("    FieldDescriptor {\n");
    out.push_str(&format!("        path: \"{}\",\n", args.path));
    out.push_str(&format!(
        "        kind: FieldType::{},\n",
        args.kind.enum_variant()
    ));
    out.push_str(&format!(
        "        reload: ReloadKind::{},\n",
        args.reload.enum_variant()
    ));
    out.push_str("        scope: SurfaceScope::Universal,\n");
    out.push_str("    },\n");
    out.push('\n');

    out.push_str("## 6. Run `cargo test -p smallaios-mgmt --tests` and the universal-exposure\n");
    out.push_str("##    walker will confirm every active surface can read/write the new field.\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_field_type_known_names() {
        assert_eq!(FieldType::parse("bool").unwrap(), FieldType::Bool);
        assert_eq!(FieldType::parse("u32").unwrap(), FieldType::U32);
        assert_eq!(FieldType::parse("u64").unwrap(), FieldType::U64);
        assert_eq!(FieldType::parse("string").unwrap(), FieldType::String);
        assert_eq!(
            FieldType::parse("password_classes").unwrap(),
            FieldType::PasswordClasses
        );
    }

    #[test]
    fn parse_field_type_rejects_unknown() {
        assert!(FieldType::parse("u128").is_err());
        assert!(FieldType::parse("").is_err());
    }

    #[test]
    fn parse_reload_known_names() {
        assert_eq!(ReloadKind::parse("live").unwrap(), ReloadKind::Live);
        assert_eq!(ReloadKind::parse("boot").unwrap(), ReloadKind::Boot);
    }

    #[test]
    fn parse_reload_rejects_unknown() {
        assert!(ReloadKind::parse("eager").is_err());
    }

    #[test]
    fn namespace_extraction_from_path() {
        let a = ScaffoldArgs {
            path: "idle.foo".into(),
            kind: FieldType::U32,
            reload: ReloadKind::Live,
        };
        assert_eq!(a.namespace(), "idle");
        assert_eq!(a.field_name(), "foo");
    }

    #[test]
    fn namespace_extraction_handles_single_segment() {
        let a = ScaffoldArgs {
            path: "scalar".into(),
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
        };
        assert_eq!(a.namespace(), "scalar");
        assert_eq!(a.field_name(), "scalar");
    }

    #[test]
    fn render_diff_includes_path() {
        let a = ScaffoldArgs {
            path: "idle.warn_minutes".into(),
            kind: FieldType::U32,
            reload: ReloadKind::Live,
        };
        let s = render_diff(&a);
        assert!(s.contains("idle.warn_minutes"));
        assert!(s.contains("FieldType::U32"));
        assert!(s.contains("ReloadKind::Live"));
    }

    #[test]
    fn render_diff_for_boot_kind() {
        let a = ScaffoldArgs {
            path: "fs.new_thing".into(),
            kind: FieldType::U64,
            reload: ReloadKind::Boot,
        };
        let s = render_diff(&a);
        assert!(s.contains("ReloadKind::Boot"));
        assert!(s.contains("FieldType::U64"));
    }

    #[test]
    fn render_diff_includes_default_placeholder() {
        let a = ScaffoldArgs {
            path: "idle.foo".into(),
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
        };
        let s = render_diff(&a);
        assert!(s.contains("CHOOSE A SAFE DEFAULT"));
    }

    #[test]
    fn render_diff_mentions_universal_exposure_walker() {
        let a = ScaffoldArgs {
            path: "x.y".into(),
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
        };
        let s = render_diff(&a);
        assert!(s.contains("universal-exposure"));
    }

    #[test]
    fn display_includes_path_and_kind() {
        let a = ScaffoldArgs {
            path: "x.y".into(),
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
        };
        let s = format!("{}", a);
        assert!(s.contains("x.y"));
        assert!(s.contains("Bool"));
    }
}
