# modules/static_manager.py
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from typing import Tuple


SECTIONS_DIRNAME = "static_sections"
TEMPLATES_DIRNAME = "doc_templates"
REGISTRY_FILENAME = "static_sections_registry.json"
SETTINGS_FILENAME = "app_settings.json"

DEFAULT_SECTIONS = [
    "Cover Letter",
    "Executive Summary",
    "Experience",
    "Offerings",
    "Team & Credentials",
    "Case Studies",
    "Professional References",
]

@dataclass
class StaticRegistry:
    sections: Dict[str, str]         # section name -> absolute path to .docx
    templates_dir: str               # absolute path
    sections_dir: str                # absolute path
    default_template: Optional[str]  # absolute path or None

def _ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    sections_dir = base_dir / SECTIONS_DIRNAME
    templates_dir = base_dir / TEMPLATES_DIRNAME
    sections_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    return sections_dir, templates_dir

def _reg_path(base_dir: Path) -> Path:
    return base_dir / REGISTRY_FILENAME

def _settings_path(base_dir: Path) -> Path:
    return base_dir / SETTINGS_FILENAME

def load_registry(base_dir: str | Path) -> StaticRegistry:
    base = Path(base_dir)
    sections_dir, templates_dir = _ensure_dirs(base)
    rp = _reg_path(base)
    if rp.exists():
        d = json.loads(rp.read_text(encoding="utf-8"))
        return StaticRegistry(
            sections=d.get("sections", {}),
            templates_dir=str(templates_dir),
            sections_dir=str(sections_dir),
            default_template=d.get("default_template"),
        )
    # create default empty registry
    reg = StaticRegistry(
        sections={name: "" for name in DEFAULT_SECTIONS},
        templates_dir=str(templates_dir),
        sections_dir=str(sections_dir),
        default_template=None,
    )
    save_registry(base, reg)
    return reg

def save_registry(base_dir: str | Path, reg: StaticRegistry):
    base = Path(base_dir)
    data = {
        "sections": reg.sections,
        "default_template": reg.default_template,
    }
    _reg_path(base).write_text(json.dumps(data, indent=2), encoding="utf-8")

def list_templates(reg: StaticRegistry) -> List[str]:
    td = Path(reg.templates_dir)
    return [str(p) for p in td.glob("*.docx")] + [str(p) for p in td.glob("*.dotx")]

def add_or_replace_section_file(reg: StaticRegistry, section_name: str, src_path: str) -> str:
    # Copy into sections_dir with stable name
    dest = Path(reg.sections_dir) / f"{section_name.replace(' ', '_')}.docx"
    shutil.copy2(src_path, dest)
    reg.sections[section_name] = str(dest)
    return str(dest)

def delete_section_file(reg: StaticRegistry, section_name: str):
    p = reg.sections.get(section_name, "")
    if p:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    reg.sections[section_name] = ""

def add_template(reg: StaticRegistry, src_path: str) -> str:
    dest = Path(reg.templates_dir) / Path(src_path).name
    shutil.copy2(src_path, dest)
    return str(dest)

def remove_template(reg: StaticRegistry, template_path: str):
    try:
        Path(template_path).unlink(missing_ok=True)
    except Exception:
        pass
    if reg.default_template and Path(reg.default_template).samefile(template_path):
        reg.default_template = None
