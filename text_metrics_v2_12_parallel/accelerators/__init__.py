"""Optional accelerator loaders for exact-result fast paths.

The production pipeline always has a pure-Python reference implementation. This
package is the narrow boundary where future compiled accelerators can be loaded
without changing the surrounding pipeline logic. When no compiled accelerator is
available, the pipeline keeps using the Python reference path.
"""
