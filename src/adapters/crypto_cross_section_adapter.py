from __future__ import annotations

from pathlib import Path

from adapters.cross_section_domain_adapter import CrossSectionDomainAdapter


class CryptoCrossSectionDomainAdapter(CrossSectionDomainAdapter):
    def __init__(
        self,
        panel_data_path: str | Path,
        *,
        text_data_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        write_artifacts: bool = False,
        debug_symbol_count: int = 20,
        debug_time_steps: int = 180,
    ) -> None:
        super().__init__(
            panel_data_path,
            market_type="crypto",
            text_data_path=text_data_path,
            artifact_dir=artifact_dir,
            write_artifacts=write_artifacts,
            debug_symbol_count=debug_symbol_count,
            debug_time_steps=debug_time_steps,
        )
