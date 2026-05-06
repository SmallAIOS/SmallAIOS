## MODIFIED Requirements

### Requirement: /models/ becomes a merged view when overlay feature is enabled
With the `fs-overlay-mounts` cargo feature enabled, the `/models/` mount point SHALL no longer be a direct squashfs mount. It SHALL instead be a merged view backed by the overlay implementation in `fs-overlay-mount`. The lower layer SHALL be the active squashfs slot per `embedded-filesystem-v1`'s A/B selection. The upper layer SHALL be `/data/models-upper/` on F2FS.

When the `fs-overlay-mounts` feature is disabled, the existing `embedded-filesystem-v1` behavior SHALL apply unchanged: `/models/` is a direct squashfs mount, all writes return `-EROFS`.

#### Scenario: Overlay-enabled merged view
- **WHEN** the kernel is built with `fs-overlay-mounts`
- **THEN** lookups under `/models/` SHALL apply upper-wins precedence
- **AND** writes to upper-only or new files SHALL succeed
- **AND** writes to lower-only files SHALL return `-EROFS` with the model_add hint

#### Scenario: Overlay-disabled compatibility
- **WHEN** the kernel is built without `fs-overlay-mounts`
- **THEN** `/models/` SHALL behave exactly as in `embedded-filesystem-v1`
- **AND** any write SHALL return `-EROFS`
- **AND** no upper-layer code SHALL execute
