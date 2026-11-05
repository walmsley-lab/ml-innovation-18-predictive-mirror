# PR 17 — Hardware-layout sensitivity mapping

An **analytic** 2D mesh mapping with per-message Manhattan hops, local
capacity and a crude offchip spill floor. Does not model Tenstorrent
execution accurately or claim validated SRAM/DRAM/NoC performance.

Use PR 04 resource traces, PR 16 active-update fractions and actual
architecture statistics to evaluate placement sensitivity at fixed
capability. Selection into hardware mapping requires a useful structural
finding or a compelling falsification question. Hardware programmes are
currently gated by their own evidence and contract requirements.
