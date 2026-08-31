"""Canonical source-first page generation for experimental V4."""

from .core import (
    CanonicalBlock,
    CanonicalPage,
    PageBuildResult,
    blocks_from_document,
    build_page_tex,
    bundle_blocks,
    pack_blocks,
    render_table_html_to_latex,
    verify_rendered_text,
)
from .mutation import (
    CONFUSABLES,
    MUTATION_POLICY_VERSION,
    MutationValidation,
    PageMutation,
    RenderedWord,
    apply_page_mutations,
    choose_page_mutations,
    validate_mutated_word_geometry,
)

__all__ = [
    "CONFUSABLES",
    "MUTATION_POLICY_VERSION",
    "CanonicalBlock",
    "CanonicalPage",
    "MutationValidation",
    "PageBuildResult",
    "PageMutation",
    "RenderedWord",
    "apply_page_mutations",
    "blocks_from_document",
    "build_page_tex",
    "bundle_blocks",
    "choose_page_mutations",
    "pack_blocks",
    "render_table_html_to_latex",
    "validate_mutated_word_geometry",
    "verify_rendered_text",
]
