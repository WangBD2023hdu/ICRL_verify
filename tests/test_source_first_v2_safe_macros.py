from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from arxiv_source_first_v2.safe_macros import (
    MacroExpansionError,
    collect_safe_macros,
    expand_safe_macros,
)


class SafeMacrosTest(unittest.TestCase):
    def _source(self, directory: str, name: str, value: str) -> Path:
        path = Path(directory) / name
        path.write_text(value, encoding="utf-8")
        return path

    def test_lipics_zero_and_one_argument_macros_expand_from_source(self) -> None:
        definitions = r"""
\usepackage{xspace}
\newcommand{\name}{\textsc{BigDipper}\xspace}
\newcommand{\DAL}{\textmd{DA-CR}\xspace}
\newcommand{\cardOne}{{\textsc{Card-}$7$}\xspace}
\newcommand{\primitive}[1]{\textsf{{#1}}}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "main.tex", definitions)
            registry = collect_safe_macros([source_path])

            self.assertEqual(
                set(registry.accepted_names),
                {"DAL", "cardOne", "name", "primitive"},
            )
            fragment = r"Use \name for $\DAL$ and {\cardOne}; call \primitive{Retrieve}."
            result = expand_safe_macros(
                fragment,
                registry,
                source_file=source_path,
                source_base_offset=500,
            )

            self.assertEqual(
                result.text,
                r"Use \textsc{BigDipper} for $\textmd{DA-CR}$ and "
                r"{{\textsc{Card-}$7$}}; call \textsf{{Retrieve}}.",
            )
            self.assertEqual(
                set(result.macros_used),
                {"DAL", "cardOne", "name", "primitive"},
            )
            name_event = next(
                item for item in result.provenance if item.macro_name == "name"
            )
            start = fragment.index(r"\name")
            # The terminating source space is a consumed TeX control-word
            # delimiter and therefore belongs to the invocation provenance.
            self.assertEqual(name_event.invocation_span, (500 + start, 500 + start + 6))
            self.assertEqual(
                result.text[slice(*name_event.output_span)],
                r"\textsc{BigDipper} ",
            )
            primitive_event = next(
                item for item in result.provenance if item.macro_name == "primitive"
            )
            argument_start = fragment.index("Retrieve")
            self.assertEqual(
                primitive_event.argument_spans,
                ((500 + argument_start, 500 + argument_start + len("Retrieve")),),
            )
            self.assertEqual(
                primitive_event.definition_body_sha256,
                registry.by_name["primitive"].body_sha256,
            )

    def test_recursive_dependency_dag_and_nested_argument_provenance(self) -> None:
        definitions = r"""
\newcommand{\base}{Core}
\newcommand{\alias}{\textbf{\base}}
\newcommand{\wrap}[1]{\alias: #1}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "defs.tex", definitions)
            registry = collect_safe_macros([source_path])
            result = expand_safe_macros(
                r"A \wrap{\base}.", registry, source_file=source_path
            )

            self.assertEqual(result.text, r"A \textbf{Core}: Core.")
            self.assertEqual(result.macros_used, ("alias", "base", "wrap"))
            self.assertGreaterEqual(result.maximum_depth, 2)
            wrap = next(item for item in result.provenance if item.macro_name == "wrap")
            self.assertEqual(result.text[slice(*wrap.output_span)], r"\textbf{Core}: Core")
            nested = [
                item
                for item in result.provenance
                if item.macro_name == "base" and len(item.expansion_stack) > 1
            ]
            self.assertTrue(nested)
            self.assertTrue(
                all(item.definition_span == registry.by_name["base"].declaration_span for item in nested)
            )

    def test_unsafe_semantics_and_non_visible_wrappers_are_rejected(self) -> None:
        definitions = r"""
\newcommand{\badif}[1]{\ifnum1=1 #1\fi}
\newcommand{\badwrite}[1]{\write18{#1}}
\newcommand{\badref}[1]{\ref{#1}}
\newcommand{\badlayout}[1]{\vspace{#1}}
\newcommand{\badcs}[1]{\csname #1\endcsname}
\newcommand{\drop}[1]{fixed}
\newcommand{\duplicate}[1]{#1#1}
\newcommand{\unknown}{\mystery}
\newcommand{\invisible}{}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "unsafe.tex", definitions)
            registry = collect_safe_macros([source_path])

            self.assertEqual(registry.accepted_names, ())
            reasons = {item.name: item.reason for item in registry.rejections}
            self.assertIn("forbidden_command:condition", reasons["badif"])
            self.assertIn("forbidden_command:io", reasons["badwrite"])
            self.assertIn("forbidden_command:label_or_key", reasons["badref"])
            self.assertIn("forbidden_command:layout", reasons["badlayout"])
            self.assertIn("forbidden_command:control_sequence_name", reasons["badcs"])
            self.assertEqual(reasons["drop"], "argument_occurrences=0:expected=1")
            self.assertEqual(
                reasons["duplicate"], "argument_occurrences=2:expected=1"
            )
            self.assertEqual(reasons["unknown"], "unknown_command:mystery")
            self.assertEqual(reasons["invisible"], "no_visible_output")

    def test_cycles_redefinitions_and_unsupported_def_are_fail_closed(self) -> None:
        definitions = r"""
\newcommand{\a}{\b}
\newcommand{\b}{\a}
\newcommand{\depends}{\a}
\newcommand{\twice}{first}
\renewcommand{\twice}{second}
\def\legacy{legacy}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "ambiguous.tex", definitions)
            registry = collect_safe_macros([source_path])
            reasons = {}
            for item in registry.rejections:
                reasons.setdefault(item.name, set()).add(item.reason)

            self.assertIn("dependency_cycle", reasons["a"])
            self.assertIn("dependency_cycle", reasons["b"])
            self.assertIn("dependency_rejected:a", reasons["depends"])
            self.assertNotIn("dependency_cycle", reasons["depends"])
            self.assertIn(
                "ambiguous_or_unsupported_redefinition", reasons["twice"]
            )
            self.assertIn("unsupported_declaration:def", reasons["legacy"])
            with self.assertRaisesRegex(
                MacroExpansionError, "macro_definition_rejected:twice"
            ):
                expand_safe_macros(r"\twice", registry)

    def test_unknown_and_forbidden_invocations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(
                temporary, "defs.tex", r"\newcommand{\ok}{visible}"
            )
            registry = collect_safe_macros([source_path])
            with self.assertRaisesRegex(MacroExpansionError, "unknown_command:nope"):
                expand_safe_macros(r"\nope", registry)
            with self.assertRaisesRegex(
                MacroExpansionError, "forbidden_command:label_or_key:ref"
            ):
                expand_safe_macros(r"\ref{secret-key}", registry)
            with self.assertRaisesRegex(MacroExpansionError, "unknown_command:alpha"):
                expand_safe_macros(r"$\alpha$", registry)
            result = expand_safe_macros(
                r"$\alpha$",
                registry,
                additional_passthrough_commands={"alpha": 0},
            )
            self.assertEqual(result.text, r"$\alpha$")

    def test_control_symbols_are_closed_and_visible_escapes_survive(self) -> None:
        definitions = """\\newcommand{\\percent}{100\\%}\n\\newcommand{\\line}{A\\\\B}\n"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "symbols.tex", definitions)
            registry = collect_safe_macros([source_path])
            self.assertIn("percent", registry.accepted_names)
            reasons = {item.name: item.reason for item in registry.rejections}
            self.assertEqual(
                reasons["line"],
                "unknown_or_nonvisible_control_symbol:'\\\\'",
            )
            self.assertEqual(
                expand_safe_macros(r"Rate: \percent.", registry).text,
                r"Rate: 100\%.",
            )
            with self.assertRaisesRegex(
                MacroExpansionError, "unknown_or_nonvisible_control_symbol"
            ):
                expand_safe_macros("A\\\\B", registry)

    def test_redefined_passthrough_and_forbidden_names_poison_dependents(self) -> None:
        definitions = r"""
\renewcommand{\textbf}[1]{#1}
\newcommand{\parent}[1]{\textbf{#1}}
\newcommand{\ref}[1]{reference #1}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "shadow.tex", definitions)
            registry = collect_safe_macros([source_path])
            reasons = {item.name: item.reason for item in registry.rejections}
            self.assertEqual(
                reasons["textbf"], "safe_passthrough_command_redefined"
            )
            self.assertEqual(reasons["parent"], "dependency_rejected:textbf")
            self.assertIn("forbidden_definition_name", reasons["ref"])
            self.assertNotIn(("textbf", 1), registry.allowed_commands)

    def test_xspace_requires_a_conservatively_known_follower(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(
                temporary,
                "defs.tex",
                r"\newcommand{\brand}{Brand\xspace}",
            )
            registry = collect_safe_macros([source_path])
            self.assertEqual(expand_safe_macros(r"\brand word", registry).text, "Brand word")
            self.assertEqual(expand_safe_macros(r"\brand, word", registry).text, "Brand, word")
            with self.assertRaisesRegex(
                MacroExpansionError, "xspace follower is ambiguous"
            ):
                expand_safe_macros(r"\brand{word}", registry)

    def test_comments_are_ignored_for_collection_but_rejected_in_bodies(self) -> None:
        definitions = """% \\newcommand{\\commented}{bad}\n\\newcommand{\\good}{Good}\n\\newcommand{\\bodycomment}{A% hidden\nB}\n"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "comments.tex", definitions)
            registry = collect_safe_macros([source_path])
            self.assertEqual(registry.accepted_names, ("good",))
            reasons = {item.name: item.reason for item in registry.rejections}
            self.assertNotIn("commented", reasons)
            self.assertEqual(reasons["bodycomment"], "comment_in_definition_body")

    def test_expansion_limits_are_enforced(self) -> None:
        definitions = r"""
\newcommand{\a}{A}
\newcommand{\b}{\a\a\a}
"""
        with tempfile.TemporaryDirectory() as temporary:
            source_path = self._source(temporary, "limits.tex", definitions)
            registry = collect_safe_macros([source_path])
            with self.assertRaisesRegex(
                MacroExpansionError, "expansion_count_exceeded"
            ):
                expand_safe_macros(r"\b", registry, max_expansions=2)
            with self.assertRaisesRegex(
                MacroExpansionError, "expanded_output_limit_exceeded"
            ):
                expand_safe_macros(r"\b", registry, max_output_characters=2)


if __name__ == "__main__":
    unittest.main()
