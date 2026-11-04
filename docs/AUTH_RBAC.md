# AeroMind login and role access

The application now uses individual user accounts, revocable bearer sessions, and server-enforced role-based access control. Passwords are stored as salted PBKDF2-SHA256 hashes; session tokens are stored only as SHA-256 hashes.

## Demo accounts

With `ENABLE_DEMO_SEED=true`, the first startup creates these accounts:

| Account | Role |
|---|---|
| `admin@aeromind.local` | Admin |
| `manager@aeromind.local` | Manager |
| `inspector@aeromind.local` | Inspector |
| `viewer@aeromind.local` | Viewer |

The default demo password is `AeroMind@123`. Set `AUTH_DEMO_PASSWORD` before the database is created to choose a different one. Existing account passwords are not overwritten at startup.

## Role policy

| Role | Access |
|---|---|
| Viewer | Read-only sites, tower details, 3D twins, inventory, findings, comparisons, and reports |
| Inspector | Viewer access plus uploads, processing, AI detection, reconstruction, QC, and review decisions |
| Manager | Inspector access plus tower/site and inspection management, deletion, and maintenance workflows |
| Admin | Full access plus users, roles, models, taxonomy, integrations, and settings |

The sidebar hides modules outside the signed-in user's role. The backend independently enforces the same policy, so a hidden page or handcrafted request cannot bypass it.

## Production setup

Set the following before first startup:

```env
ENABLE_DEMO_SEED=false
AUTH_SHOW_DEMO_CREDENTIALS=false
AUTH_ADMIN_EMAIL=admin@example.com
AUTH_ADMIN_NAME=AeroMind Admin
AUTH_ADMIN_PASSWORD=use-a-long-random-password
```

`AUTH_ADMIN_PASSWORD` is required when demo seeding is disabled. `AUTH_DISABLED=true` is intended only for isolated automated tests and must not be used in a deployed environment.

Admins manage accounts from **Administration → Users & Roles**. They can add users, change roles, deactivate access, and reset passwords. Password resets revoke all existing sessions for that user.
