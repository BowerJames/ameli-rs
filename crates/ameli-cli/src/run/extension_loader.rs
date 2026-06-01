//! Dynamic extension loading from compiled dylibs.
//!
//! Loads extensions from shared library files (`.so` / `.dylib` / `.dll`)
//! using [`libloading`]. Each library must export a well-known entry point
//! that returns a double-boxed `Box<dyn Extension>` trait object.
//!
//! # FFI Convention
//!
//! The dylib must export:
//!
//! ```ignore
//! #[no_mangle]
//! pub extern "C" fn ameli_create_extension() -> *mut () {
//!     let ext: Box<dyn ameli_agent::extension::Extension> = Box::new(MyExtension);
//!     // Double-box: `Box<dyn Extension>` is a fat pointer. Boxing it yields
//!     // a thin pointer that can safely cross the FFI boundary.
//!     Box::into_raw(Box::new(ext)) as *mut ()
//! }
//! ```
//!
//! # Safety
//!
//! - The `Library` handle must outlive any trait objects obtained from it.
//!   [`ExtensionSet`] enforces this by keeping both in the same struct.
//! - The dylib must be compiled with the same Rust toolchain and same
//!   `ameli-agent` dependency version as the CLI. This is a well-known Rust
//!   limitation (no stable ABI).

use ameli_agent::extension::Extension;
use anyhow::{bail, Context, Result};
use libloading::Library;
use std::path::Path;

// ---------------------------------------------------------------------------
// ExtensionSet — keeps libraries and extensions alive together
// ---------------------------------------------------------------------------

/// A set of extensions loaded from dynamic libraries.
///
/// The `Library` handles are stored alongside the extensions to ensure the
/// dylib mappings (and therefore vtables) remain valid for the lifetime of
/// the extensions. Dropping this struct unloads the libraries.
pub struct ExtensionSet {
    /// The loaded extensions, ready to pass to
    /// [`create_agent_session`](ameli_agent::create_agent_session).
    pub extensions: Vec<Box<dyn Extension>>,
    /// Library handles — kept alive so extension vtables remain valid.
    /// The underscore prefix signals these are deliberately kept for
    /// lifetime purposes.
    pub _libraries: Vec<Library>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load all extensions from the given file paths.
///
/// Each file must be a compiled dynamic library exporting the
/// `ameli_create_extension` entry point (see module-level docs).
///
/// # Errors
///
/// Returns an error if any library cannot be loaded, the entry point symbol
/// is missing, or the entry point returns null.
pub fn load_extension_set(paths: &[String]) -> Result<ExtensionSet> {
    let mut extensions: Vec<Box<dyn Extension>> = Vec::new();
    let mut libraries: Vec<Library> = Vec::new();

    for path in paths {
        let path = path.trim();
        if path.is_empty() {
            bail!("extension path must not be empty");
        }
        let (ext, lib) = load_one(path)?;
        extensions.push(ext);
        libraries.push(lib);
    }

    Ok(ExtensionSet {
        extensions,
        _libraries: libraries,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Load a single extension from a dylib path.
fn load_one(path: &str) -> Result<(Box<dyn Extension>, Library)> {
    if !Path::new(path).exists() {
        bail!("extension file not found: {path}");
    }

    // SAFETY: Loading a shared library via `Library::new` is unsafe because
    // the library could execute arbitrary code in its constructor. This is
    // inherent to dynamic loading. We trust the user-provided path.
    let lib = unsafe { Library::new(path) }
        .with_context(|| format!("failed to load extension library: {path}"))?;

    // SAFETY: `get` is unsafe because the symbol could have any type. We
    // specify the correct function signature matching the convention.
    let create_fn: libloading::Symbol<'_, unsafe extern "C" fn() -> *mut ()> = unsafe {
        lib.get(b"ameli_create_extension\0")
            .with_context(|| format!("symbol 'ameli_create_extension' not found in {path}"))?
    };

    // SAFETY: Calling the entry point is unsafe because we trust the library
    // to follow the convention (return a valid double-boxed pointer or null).
    let ptr = unsafe { create_fn() };
    if ptr.is_null() {
        bail!("ameli_create_extension returned null in {path}");
    }

    // SAFETY: We reconstruct the double box that was created by
    // `Box::into_raw(Box::new(Box::new(ext) as Box<dyn Extension>))` in the
    // dylib. The pointer is non-null (checked above) and was allocated by the
    // system allocator (both sides use the standard allocator). We immediately
    // extract the inner box and drop the outer allocation.
    let extension: Box<dyn Extension> = unsafe {
        let outer: Box<Box<dyn Extension>> = Box::from_raw(ptr as *mut Box<dyn Extension>);
        *outer
    };

    Ok((extension, lib))
}
