## ADDED Requirements

### Requirement: Network diagnostic commands

The console shell SHALL gain small read-mostly network diagnostic commands: `ip` (interface addresses), `ip route` (route table), `ip link` (link state), and `bond` (bond status).

#### Scenario: ip lists interface addresses

- **WHEN** a logged-in operator runs `ip`
- **THEN** the output SHALL list each interface with its configured IPv4 and IPv6 addresses

#### Scenario: ip route dumps the route table

- **WHEN** a logged-in operator runs `ip route`
- **THEN** the output SHALL list the route-table entries with prefix, egress interface, and metric

#### Scenario: ip link shows link state

- **WHEN** a logged-in operator runs `ip link`
- **THEN** the output SHALL show each interface's link state (up or down)

#### Scenario: bond shows mode and slave state

- **WHEN** a logged-in operator runs `bond` on a unit with `bond0` configured
- **THEN** the output SHALL show the bond's mode and each slave with its membership and link state
