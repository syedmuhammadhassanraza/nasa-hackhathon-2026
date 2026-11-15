# Security and Authentication Limitations

This project is an educational research prototype. It is not a secure clinical
system, does not implement user authentication or role-based access control,
and must not be used to store or process real astronaut or patient records.

## Current behavior

- The HTTP server binds to `127.0.0.1` by default. This limits the listener to
  the local machine; it does not provide authentication or protect against
  other processes or users who can access that machine.
- The dashboard uses third-party CDN assets and is not an air-gapped
  application.
- SQLite is not encrypted by this project. The local database can contain
  demo telemetry and scan records; protect or remove it when sharing a
  machine.
- The API does not enforce user identities, roles, or clinical permissions.
- No TLS, audit-log integrity signatures, or HIPAA/NASA security compliance
  controls are implemented.
- Camera frames are processed in the browser; the experimental scanner sends
  RGB summaries to the local server. This is not a validated clinical sensor.
- Raw datasets, generated health-data files, local databases, and virtual
  environments are excluded from the public Git repository. This exclusion
  does not make the running application suitable for real medical data.

## Before any real deployment

Do not expose this development server to a network or use it for clinical,
flight, or other safety-critical decisions. A production system would need
independent security and clinical validation, threat modeling, authenticated
and authorized access, protected transport and storage, privacy review,
operational monitoring, and mission-approved procedures. This repository
does not provide those controls.
