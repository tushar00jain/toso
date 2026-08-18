# Sensor, environment, and selector flow

<!-- Generated from sensor_environment_selector_flow.diagram.xml by realsim.tools.text_diagram. -->

## 1. Who holds what

<!-- text-diagram:who-holds-what:start -->
```
═══ the object itself
╌╌╌ a handle to a service

CONTROL (one capability)                        SEAMS                        DATA HOST × N
┌────────────────────────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────┐
│ ControlPlane                               │◄─│ ControlPlaneService     │◄╌│ .control_plane_handle         │
│                                            │  │ behind a                │  │ DataPlane                     │
│ selector chains ═══ declared Sensor types  │  │ LocalControlPlaneHandle │  │   deployment ═ Simulation     │
│ Sensing attach ═══ Environment + Sensors   │  │                         │  │                               │
│ Dispatcher ═ reducer refs to same Sensor(s)│◄─│ DispatcherService       │◄╌│ .dispatcher_handle            │
│                                            │  │ behind a                │  │ client / volume calls         │
│ dedup: FanoutSensor as fanout + load       │  │ LocalDispatcherHandle   │  │ go through Deployment         │
│ KV: Cluster, Reservation, RoutedPull,      │  │                         │  │                               │
│     SourceLoad                             │  │                         │  │ holds no control objects      │
└────────────────────────────────────────────┘  └─────────────────────────┘  └───────────────────────────────┘
                        ▲
                        │ attach(Environment, Sensors)
┌────────────────────────────── Simulation = Deployment ───────────────────────────────┐
│                                                                                      │
│ Simulation ═══ Mesh                                                                  │
│            ═══ control_plane_handle                                                  │
│            ═══ dispatcher_handle                                                     │
│                                                                                      │
│ Environment ═══ topology + MachineProfile + clock                                    │
│ DirectorySensor ═══ ControllerService                                                │
│                     └── locate / locate_live / pinned                                │
│                                                                                      │
│ Mesh ═══ ControllerService      the real directory: key → current volume holders     │
│      ═══ LocalClient × N        get / put / get_batch / put_batch                    │
│      ═══ VolumeService × N      resident bytes, capacity, eviction                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:who-holds-what:end -->

The control plane and its selectors resolve the same sensor objects by declared type.
Selectors read them; Dispatcher reducer bindings are the write side.

## 2. One generic request lifecycle

<!-- text-diagram:request-lifecycle:start -->
```
ONE REQUEST — dedup and KV-cache

┌──────────────────────────────── CONTROL PLANE ─────────────────────────────────┐
│ 4. decide   selector.select(subject, requester)                                │
│                                                                                │
│      dedup: keys → ranked source volumes + readiness                           │
│      KV:    request → prefill/decode placement + priced reuse;                 │
│             fetch keys → the source already priced                             │
│                                                                                │
│    the Selector reads declared Sensors and writes nothing                      │
│    the plane chooses the answer and commits the decision                       │
└────────────────────────────────────────────────────────────────────────────────┘
            ▲ 3. sense                                       │ 5. answer
            │                                                ▼
┌────────────── ENVIRONMENT + SENSORS ───────────────┐   ┌────────── DATA PLANE ──────────┐
│ DIRECTORY SENSOR                                   │   │ 1. request arrives             │
│   locate(keys): key → current holders              │   │ 2. ask through control handle  │
│ ENVIRONMENT                                        │   │ 5. receive the answer          │
│   topology / read time / now                       │   │ 6. actuate it                  │
│ CAPABILITY SENSOR READS                            │   │ 7. store calls move bytes      │
│   dedup: fan-out tree, owed puts, source load      │   │ 8. compute, if any             │
│   KV: predicted queues, decode batches,            │   │                                │
│       reservations, routed pulls, source load      │   │ dedup: preferred get → put     │
│                                                    │   │ KV: route → fetch/reuse        │
│ 10. the next decision reads both updated truths    │   │     → prefill → publish        │
└────────────────────────────────────────────────────┘   │     → decode                   │
                                                         │ 9. publish and report facts    │
                                                         └────────────────────────────────┘
            ▲ store put / eviction    ▲ reported Action
            │                         │
┌───── DIRECTORY TRUTH ──────┐   ┌───────── DISPATCHER ──────────┐
│ put / put_batch registers  │   │ one Action folds into every   │
│ eviction deletes holders   │   │ affected Sensor, then commits │
└────────────────────────────┘   └───────────────────────────────┘
```
<!-- text-diagram:request-lifecycle:end -->

Dedup feedback: the batch put changes the directory; `Published` settles its promise.
KV feedback: publish/evict changes prefix presence; `PrefillFinished`, `ComputeBusy`,
`DecodeState`, `Committed`, and `FetchAnswered` update the sensors for the next request.
