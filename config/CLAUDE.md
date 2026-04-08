# config/

Runtime configuration for the CoManDO planner.

- `planner_runtime_config.hpp` – Central header that defines the `PlannerRuntimeConfig` class, parsing command‑line arguments and loading YAML configuration files.
- `planner_runtime_config_types.hpp` – Type definitions for configuration parameters (e.g., cost weights, horizon lengths, solver options).

These files are included by the planner node and MPC components to configure behavior at launch time.
