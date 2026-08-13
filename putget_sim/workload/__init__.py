"""What is simulated: one synchronized read burst over a seeded key.

``put_get.py`` is the whole scenario -- the capability-free fixture, built on the
real client/controller/transport that ``realsim`` provides. ``dedup_sim`` imports
it unchanged and installs a selector, so the routed and unrouted runs are the same
workload.
"""
