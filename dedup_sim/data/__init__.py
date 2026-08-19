"""The dedup data plane: the read-through write that makes the routing true.

TorchStore's ``LocalClient.get_batch`` moves the bytes and assembles tensor slices. This
plane applies a source preference, then stores what was fetched into the reader's own
volume, then reports the completed batch. The put registers the reader in the real
directory before the report releases the next reader's withheld answer.
"""
