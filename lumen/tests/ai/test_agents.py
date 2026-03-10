import json

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import panel as pn
import pytest

from lumen.ai.agents.document_list import DocumentListAgent
from lumen.ai.schemas import Column, DocumentChunk, TableCatalogEntry

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.pane import Markdown

from lumen.ai.agents import (
    AnalysisAgent, ChatAgent, SQLAgent, VegaLiteAgent,
)
from lumen.ai.agents.analysis import make_analysis_model
from lumen.ai.agents.deck_gl import DeckGLAgent, DeckGLSpec
from lumen.ai.agents.hvplot import hvPlotAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.agents.table_list import TableListAgent
from lumen.ai.agents.validation import QueryCompletionValidation, ValidationAgent
from lumen.ai.agents.vega_lite import VegaLiteSpec, VegaLiteSpecUpdate
from lumen.ai.analysis import Analysis
from lumen.ai.editors import (
    AnalysisOutput, DeckGLEditor, LumenEditor, SQLEditor, VegaLiteEditor,
)
from lumen.ai.llm import Llm
from lumen.ai.schemas import Metaset, get_metaset
from lumen.config import SOURCE_TABLE_SEPARATOR, dump_yaml
from lumen.pipeline import Pipeline
from lumen.sources.duckdb import DuckDBSource
from lumen.views import Panel, hvPlotUIView
from lumen.views.base import DeckGLView

root = str(Path(__file__).parent.parent / "sources")

@pytest.fixture
def duckdb_source():
    duckdb_source = DuckDBSource(
        initializers=["INSTALL sqlite;", "LOAD sqlite;", f"SET home_directory='{root}';"],
        root=root,
        tables={"test_sql": f"SELECT A, B, C, D::TIMESTAMP_NS AS D FROM READ_CSV('{root + '/test.csv'}')"},
    )
    return duckdb_source


@pytest.fixture
def test_messages():
    """Create test messages for agent respond method"""
    return [{"role": "user", "content": "Test message"}]

async def test_chat_agent(llm, test_messages):
    agent = ChatAgent(llm=llm)

    llm.set_responses([
        "Test Response"
    ])

    out, out_context = await agent.respond(test_messages, {})
    assert out[0].object == "Test Response"

async def test_chat_agent_with_data(llm, duckdb_source, test_messages):
    """Test ChatAgent in analyst mode (with data)"""
    agent = ChatAgent(llm=llm)
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "data": [{"A": 1, "B": 2}],
        "sql": "SELECT * FROM test_sql"
    }
    llm.set_responses([
        "Analysis of data"
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert out[0].object == "Analysis of data"
    assert out_context == {}

async def test_sql_agent(llm, duckdb_source, test_messages):
    agent = SQLAgent(llm=llm)

    context = {
        "source": duckdb_source,
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
    }
    # Create the proper SQL model with tables field for single source
    SQLQueryWithTables = make_sql_model([(duckdb_source.name, "test_sql")])
    llm.set_responses([
        SQLQueryWithTables(
            query="SELECT SUM(A) as A_sum FROM test_sql",
            table_slug="test_sql_agg",
            tables=["test_sql"]
        ),
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], SQLEditor)
    assert out[0].spec == (
        "SELECT\n"
        "  SUM(A) AS A_sum\n"
        "FROM test_sql"
    )
    assert set(out_context) == {"data", "pipeline", "sql", "table", "source"}

async def test_vegalite_agent(llm, duckdb_source, test_messages):
    """Test VegaLiteAgent instantiation and respond"""

    agent = VegaLiteAgent(llm=llm, code_execution="disabled")

    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql"),
        "table": "test_sql",
        "sources": [duckdb_source],
        "metaset": await get_metaset([duckdb_source], ["test_sql"]),
        "data": duckdb_source.get("test_sql")
    }

    spec = {
        "data": {
            "values": [
                {"A": 1, "B": 2, "C": 3, "D": "2023-01-01T00:00:00Z"},
                {"A": 4, "B": 5, "C": 6, "D": "2023-01-02T00:00:00Z"},
            ]
        },
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"},
        }
    }
    llm.set_responses([
        VegaLiteSpec(
            chain_of_thought="Test plot",
            yaml_spec=dump_yaml(spec),
            insufficient_context=False,
            insufficient_context_reason="none"
        ),
        VegaLiteSpecUpdate(
            chain_of_thought="All good",
            yaml_update=""
        )
    ])
    out, out_context = await agent.respond(test_messages, context)
    assert len(out) == 1
    assert isinstance(out[0], VegaLiteEditor)
    assert out[0].spec == "$schema: https://vega.github.io/schema/vega-lite/v5.json\ndata:\n  values:\n  - A: 1\n    B: 2\n    C: 3\n    D: '2023-01-01T00:00:00Z'\n  - A: 4\n    B: 5\n    C: 6\n    D: '2023-01-02T00:00:00Z'\nencoding:\n  x:\n    field: A\n    type: quantitative\n  y:\n    field: B\n    type: quantitative\nheight: container\nmark: bar\nwidth: container\n"


async def test_analysis_agent(llm, duckdb_source, test_messages):

    class TestAnalysis(Analysis):

        def __call__(self, pipeline, context):
            return f"Test Analysis"

    agent = AnalysisAgent(
        analyses=[TestAnalysis.instance(name='foo'), TestAnalysis.instance(name='bar')],
        llm=llm
    )
    context = {
        "source": duckdb_source,
        "pipeline": Pipeline(source=duckdb_source, table="test_sql")
    }

    model = make_analysis_model([analysis.name for analysis in agent.analyses])
    llm.set_responses([
        model(analysis="bar")
    ])
    out, out_context = await agent.respond(test_messages, context)

    assert len(out) == 1
    assert isinstance(out[0], AnalysisOutput)
    assert isinstance(out[0].component, Panel)
    assert isinstance(out[0].component.object, Markdown)

    assert "view" in out_context
    assert out_context["view"]["type"] == "panel"
    assert out_context["view"]["object"]["object"] == "Test Analysis"



@pytest.mark.asyncio
class TestDocumentListAgentIntegration:
    """Tests for DocumentListAgent with metaset."""

    async def test_document_list_agent_with_metaset(self):
        """Test that DocumentListAgent works with metaset.docs."""
        # Create metaset with document chunks
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[
                DocumentChunk(filename="readme.md", text="chunk 1", similarity=0.9),
                DocumentChunk(filename="readme.md", text="chunk 2", similarity=0.8),
                DocumentChunk(filename="schema.md", text="chunk 3", similarity=0.7),
            ]
        )
        
        context = {"metaset": metaset}
        
        # Test applies
        applies = await DocumentListAgent.applies(context)
        assert applies is True  # More than 1 unique document
        
        # Test _get_items
        agent = DocumentListAgent()
        items = agent._get_items(context)
        
        # Should return unique, sorted filenames
        assert items == {"Documents": ["readme.md", "schema.md"]}

    async def test_document_list_agent_no_docs(self):
        """Test that DocumentListAgent doesn't apply when no docs."""
        # Metaset without docs
        metaset = Metaset(query="test", catalog={}, docs=None)
        context = {"metaset": metaset}
        
        applies = await DocumentListAgent.applies(context)
        assert applies is False

    async def test_document_list_agent_single_doc(self):
        """Test that DocumentListAgent doesn't apply for single doc."""
        # Metaset with only one unique document
        metaset = Metaset(
            query="test",
            catalog={},
            docs=[DocumentChunk(filename="readme.md", text="chunk", similarity=0.9)]
        )
        context = {"metaset": metaset}

        applies = await DocumentListAgent.applies(context)
        assert applies is True


# ---------------------------------------------------------------------------
# hvPlotAgent tests
# ---------------------------------------------------------------------------

class TestHvPlotAgent:

    def test_instantiation(self, llm):
        """hvPlotAgent can be created with default params."""
        agent = hvPlotAgent(llm=llm)
        assert agent.view_type is hvPlotUIView
        assert agent.purpose

    def test_conditions(self, llm):
        """Conditions mention exploratory analysis."""
        agent = hvPlotAgent(llm=llm)
        assert len(agent.conditions) >= 1
        assert any("exploratory" in c.lower() or "eda" in c.lower() for c in agent.conditions)

    @pytest.mark.xfail(reason="param_to_pydantic does not handle param.Range (xlim/ylim) — BUG in translate.py")
    def test_get_model_returns_pydantic(self, llm):
        """_get_model produces a Pydantic model with chain_of_thought and kind."""
        agent = hvPlotAgent(llm=llm)
        schema = {"A": {"type": "integer"}, "B": {"type": "integer"}}
        model = agent._get_model("main", schema)
        field_names = set(model.model_fields.keys())
        assert "chain_of_thought" in field_names
        assert "kind" in field_names

    @pytest.mark.xfail(reason="param_to_pydantic does not handle param.Range (xlim/ylim) — BUG in translate.py")
    def test_get_model_excludes_internal_params(self, llm):
        """Internal/source/pipeline params should be excluded from the model."""
        agent = hvPlotAgent(llm=llm)
        schema = {"x": {"type": "string"}}
        model = agent._get_model("main", schema)
        field_names = set(model.model_fields.keys())
        for excluded in ("controls", "source", "pipeline", "transforms", "type"):
            assert excluded not in field_names

    async def test_extract_spec_adds_defaults(self, llm, duckdb_source):
        """_extract_spec should add responsive=True."""
        agent = hvPlotAgent(llm=llm)
        pipeline = Pipeline(source=duckdb_source, table="test_sql")
        context = {"pipeline": pipeline}
        raw_spec = {"kind": "scatter", "x": "A", "y": "B"}
        result = await agent._extract_spec(context, raw_spec)
        assert result["responsive"] is True
        assert "type" not in result

    async def test_extract_spec_rasterize_large_data(self, llm):
        """Large datasets (>20k) with scatter/line/points get rasterize=True."""
        agent = hvPlotAgent(llm=llm)
        large_source = DuckDBSource(tables={
            "big": "SELECT i AS x, i * 2 AS y FROM generate_series(0, 25000) t(i)"
        })
        pipeline = Pipeline(source=large_source, table="big")
        context = {"pipeline": pipeline}

        raw_spec = {"kind": "scatter", "x": "x", "y": "y"}
        result = await agent._extract_spec(context, raw_spec)
        assert result["rasterize"] is True
        assert result["cnorm"] == "log"

    async def test_extract_spec_no_rasterize_small_data(self, llm, duckdb_source):
        """Small datasets should not get rasterize."""
        agent = hvPlotAgent(llm=llm)
        pipeline = Pipeline(source=duckdb_source, table="test_sql")
        context = {"pipeline": pipeline}

        raw_spec = {"kind": "scatter", "x": "A", "y": "B"}
        result = await agent._extract_spec(context, raw_spec)
        assert "rasterize" not in result

    async def test_extract_spec_no_rasterize_bar_chart(self, llm):
        """Bar charts should not get rasterized even with large data."""
        agent = hvPlotAgent(llm=llm)
        large_source = DuckDBSource(tables={
            "big": "SELECT i AS x, i * 2 AS y FROM generate_series(0, 25000) t(i)"
        })
        pipeline = Pipeline(source=large_source, table="big")
        context = {"pipeline": pipeline}

        raw_spec = {"kind": "bar", "x": "x", "y": "y"}
        result = await agent._extract_spec(context, raw_spec)
        assert "rasterize" not in result

    async def test_extract_spec_strips_none_values(self, llm, duckdb_source):
        """None values should be filtered out of the spec."""
        agent = hvPlotAgent(llm=llm)
        pipeline = Pipeline(source=duckdb_source, table="test_sql")
        context = {"pipeline": pipeline}
        raw_spec = {"kind": "scatter", "x": "A", "y": "B", "color": None, "size": None}
        result = await agent._extract_spec(context, raw_spec)
        assert "color" not in result
        assert "size" not in result

    @pytest.mark.xfail(reason="param_to_pydantic does not handle param.Range (xlim/ylim) — BUG in translate.py")
    async def test_respond(self, llm, duckdb_source, test_messages):
        """Full respond flow produces a LumenEditor with hvPlotUIView."""
        agent = hvPlotAgent(llm=llm)
        pipeline = Pipeline(source=duckdb_source, table="test_sql")
        context = {
            "source": duckdb_source,
            "pipeline": pipeline,
            "table": "test_sql",
            "sources": [duckdb_source],
            "metaset": await get_metaset([duckdb_source], ["test_sql"]),
            "data": duckdb_source.get("test_sql"),
        }

        schema = {"A": {"type": "integer"}, "B": {"type": "integer"}}
        model = agent._get_model("main", schema)

        llm.set_responses([
            model(chain_of_thought="Scatter of A vs B", kind="scatter", x="A", y="B"),
        ])
        out, out_context = await agent.respond(test_messages, context)
        assert len(out) == 1
        assert isinstance(out[0], LumenEditor)
        assert isinstance(out[0].component, hvPlotUIView)
        assert "view" in out_context

    async def test_respond_no_pipeline_raises(self, llm, test_messages):
        """Respond raises ValueError when no pipeline in context."""
        agent = hvPlotAgent(llm=llm)
        with pytest.raises(ValueError, match="pipeline"):
            await agent.respond(test_messages, {})


# ---------------------------------------------------------------------------
# DeckGLAgent tests
# ---------------------------------------------------------------------------

class TestDeckGLAgent:

    def test_instantiation(self, llm):
        """DeckGLAgent can be created with default params."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        assert agent.view_type is DeckGLView
        assert agent._editor_type is DeckGLEditor
        assert "deckgl" in agent._extensions

    def test_default_map_style_is_carto(self, llm):
        """Default map style should be CartoDB (no API key needed)."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        assert "cartocdn" in agent.default_map_style

    def test_conditions_mention_geographic(self, llm):
        """Conditions should mention geographic / 3D / DeckGL."""
        agent = DeckGLAgent(llm=llm)
        conditions_text = " ".join(agent.conditions).lower()
        assert "geographic" in conditions_text or "3d" in conditions_text

    def test_code_execution_modes(self, llm):
        """All code execution modes should be accepted."""
        for mode in ("disabled", "prompt", "llm", "allow"):
            agent = DeckGLAgent(llm=llm, code_execution=mode)
            assert agent.code_execution == mode
        assert DeckGLAgent(llm=llm, code_execution="disabled").code_execution_enabled is False
        assert DeckGLAgent(llm=llm, code_execution="allow").code_execution_enabled is True

    async def test_extract_spec_valid(self, llm):
        """_extract_spec converts JSON spec string into validated dict."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        deckgl_json = json.dumps({
            "initialViewState": {
                "latitude": 37.7749, "longitude": -122.4194,
                "zoom": 10, "pitch": 45, "bearing": 0,
            },
            "layers": [{
                "@@type": "ScatterplotLayer",
                "getPosition": "@@=[longitude, latitude]",
                "getRadius": 100,
            }],
        })
        result = await agent._extract_spec({}, {"json_spec": deckgl_json})
        assert "spec" in result
        assert result["sizing_mode"] == "stretch_both"
        assert result["min_height"] == 400
        for layer in result["spec"]["layers"]:
            assert "data" not in layer

    async def test_extract_spec_replaces_mapbox_style(self, llm):
        """Mapbox styles should be replaced with CartoDB."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        spec_dict = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": [{"@@type": "ScatterplotLayer"}],
            "mapStyle": "mapbox://styles/mapbox/dark-v10",
        }
        result = await agent._extract_spec({}, {"json_spec": json.dumps(spec_dict)})
        assert "mapbox" not in result["spec"]["mapStyle"]
        assert "cartocdn" in result["spec"]["mapStyle"]

    async def test_extract_spec_preserves_custom_map_style(self, llm):
        """Custom non-Mapbox styles should be preserved."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        custom_style = "https://my-tiles.example.com/style.json"
        spec_dict = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": [{"@@type": "HexagonLayer"}],
            "mapStyle": custom_style,
        }
        result = await agent._extract_spec({}, {"json_spec": json.dumps(spec_dict)})
        assert result["spec"]["mapStyle"] == custom_style

    async def test_extract_spec_strips_data_from_layers(self, llm):
        """Layer data must be removed — it's injected at render time."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        spec_dict = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": [{
                "@@type": "ScatterplotLayer",
                "data": [{"lat": 1, "lng": 2}],
                "layer_data": "extra",
            }],
        }
        result = await agent._extract_spec({}, {"json_spec": json.dumps(spec_dict)})
        layer = result["spec"]["layers"][0]
        assert "data" not in layer
        assert "layer_data" not in layer

    async def test_extract_spec_extracts_tooltips(self, llm):
        """Tooltip config should be extracted from spec."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        spec_dict = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": [{"@@type": "ScatterplotLayer"}],
            "tooltip": {"html": "<b>{name}</b>"},
        }
        result = await agent._extract_spec({}, {"json_spec": json.dumps(spec_dict)})
        assert result["tooltips"] == {"html": "<b>{name}</b>"}
        assert "tooltip" not in result["spec"]

    async def test_extract_spec_default_tooltips(self, llm):
        """When no tooltip in spec, default to True."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        spec_dict = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": [{"@@type": "ScatterplotLayer"}],
        }
        result = await agent._extract_spec({}, {"json_spec": json.dumps(spec_dict)})
        assert result["tooltips"] is True

    async def test_extract_spec_missing_keys_raises(self, llm):
        """Missing required keys should raise ValueError."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        bad_spec = {"initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0}}
        with pytest.raises(ValueError, match="missing required keys"):
            await agent._extract_spec({}, {"json_spec": json.dumps(bad_spec)})

    async def test_extract_spec_invalid_layers_raises(self, llm):
        """Non-list layers should raise an error."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        bad_spec = {
            "initialViewState": {"latitude": 0, "longitude": 0, "zoom": 1, "pitch": 0, "bearing": 0},
            "layers": "not-a-list",
        }
        with pytest.raises((ValueError, AttributeError, TypeError)):
            await agent._extract_spec({}, {"json_spec": json.dumps(bad_spec)})

    async def test_respond_declarative_mode(self, llm, test_messages):
        """Full respond flow in declarative (code_execution=disabled) mode."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")

        geo_source = DuckDBSource(tables={
            "locations": "SELECT 37.7749 AS latitude, -122.4194 AS longitude, 'SF' AS city"
        })
        pipeline = Pipeline(source=geo_source, table="locations")
        context = {
            "source": geo_source,
            "pipeline": pipeline,
            "table": "locations",
            "sources": [geo_source],
            "metaset": await get_metaset([geo_source], ["locations"]),
            "data": geo_source.get("locations"),
        }

        deckgl_json = json.dumps({
            "initialViewState": {
                "latitude": 37.7749, "longitude": -122.4194,
                "zoom": 10, "pitch": 45, "bearing": 0,
            },
            "layers": [{
                "@@type": "ScatterplotLayer",
                "getPosition": "@@=[longitude, latitude]",
                "getRadius": 500,
                "getFillColor": [255, 0, 0],
            }],
        })
        llm.set_responses([
            DeckGLSpec(
                chain_of_thought="Plotting city locations on a 3D map",
                json_spec=deckgl_json,
                insufficient_context=False,
                insufficient_context_reason="",
            ),
        ])
        out, out_context = await agent.respond(test_messages, context)
        assert len(out) == 1
        assert isinstance(out[0], DeckGLEditor)
        assert isinstance(out[0].component, DeckGLView)
        assert "view" in out_context

    async def test_respond_no_pipeline_raises(self, llm, test_messages):
        """Respond raises ValueError when no pipeline in context."""
        agent = DeckGLAgent(llm=llm, code_execution="disabled")
        with pytest.raises(ValueError, match="pipeline"):
            await agent.respond(test_messages, {})


# ---------------------------------------------------------------------------
# ValidationAgent tests
# ---------------------------------------------------------------------------

class TestValidationAgent:

    def test_instantiation(self, llm):
        """ValidationAgent can be created with default params."""
        agent = ValidationAgent(llm=llm)
        assert agent.purpose
        assert len(agent.conditions) >= 1

    def test_conditions_exclude_data_analysis(self, llm):
        """Conditions should explicitly exclude data analysis tasks."""
        agent = ValidationAgent(llm=llm)
        conditions_text = " ".join(agent.conditions).lower()
        assert "not" in conditions_text
        assert "validate" in conditions_text

    def test_query_completion_model_correct(self):
        """QueryCompletionValidation model with correct=True."""
        result = QueryCompletionValidation(
            chain_of_thought="The query was answered correctly.",
            missing_elements=[],
            suggestions=[],
            correct=True,
        )
        assert result.correct is True
        assert result.missing_elements == []

    def test_query_completion_model_incomplete(self):
        """QueryCompletionValidation model with missing elements."""
        result = QueryCompletionValidation(
            chain_of_thought="Partially answered.",
            missing_elements=["Missing trend", "Missing comparison"],
            suggestions=["Add time series", "Compare years"],
            correct=False,
        )
        assert result.correct is False
        assert len(result.missing_elements) == 2
        assert len(result.suggestions) == 2

    def test_query_completion_model_defaults(self):
        """missing_elements and suggestions default to empty lists."""
        result = QueryCompletionValidation(
            chain_of_thought="OK",
            correct=True,
        )
        assert result.missing_elements == []
        assert result.suggestions == []

    async def test_respond_correct_validation(self, llm, test_messages):
        """When validation is correct, return result directly."""
        agent = ValidationAgent(llm=llm)
        context = {
            "data": [{"A": 1}],
            "sql": "SELECT * FROM test",
        }

        llm.set_responses([
            QueryCompletionValidation(
                chain_of_thought="Query fully answered.",
                missing_elements=[],
                suggestions=[],
                correct=True,
            ),
        ])
        out, out_context = await agent.respond(test_messages, context)
        assert len(out) == 1
        assert out[0].correct is True
        assert "validation_result" in out_context
        assert out_context["validation_result"].correct is True

    async def test_respond_incomplete_validation(self, llm, test_messages):
        """When validation fails, return result with missing elements."""
        agent = ValidationAgent(llm=llm)
        agent.interface = MagicMock(spec=pn.chat.ChatFeed)
        context = {
            "data": [{"A": 1}],
            "sql": "SELECT * FROM test",
        }

        llm.set_responses([
            QueryCompletionValidation(
                chain_of_thought="The query missed the trend analysis.",
                missing_elements=["Trend analysis"],
                suggestions=["Add a time series chart"],
                correct=False,
            ),
        ])
        out, out_context = await agent.respond(test_messages, context)
        assert len(out) == 1
        assert out[0].correct is False
        assert out[0].missing_elements == ["Trend analysis"]
        assert "validation_result" in out_context
        agent.interface.stream.assert_called_once()

    async def test_respond_with_view_context(self, llm, test_messages):
        """ValidationAgent should accept view context."""
        agent = ValidationAgent(llm=llm)
        context = {
            "data": [{"x": 1}],
            "view": {"type": "vegalite"},
        }

        llm.set_responses([
            QueryCompletionValidation(
                chain_of_thought="All steps executed successfully.",
                missing_elements=[],
                suggestions=[],
                correct=True,
            ),
        ])
        out, _ = await agent.respond(test_messages, context)
        assert out[0].correct is True

    async def test_respond_empty_context(self, llm, test_messages):
        """ValidationAgent works even with minimal context."""
        agent = ValidationAgent(llm=llm)
        agent.interface = MagicMock(spec=pn.chat.ChatFeed)

        llm.set_responses([
            QueryCompletionValidation(
                chain_of_thought="No data or views present.",
                missing_elements=["No results produced"],
                suggestions=["Run a SQL query first"],
                correct=False,
            ),
        ])
        out, out_context = await agent.respond(test_messages, {})
        assert out[0].correct is False
        assert len(out[0].suggestions) > 0

    async def test_respond_multiple_missing_elements(self, llm, test_messages):
        """Validation result correctly captures multiple missing elements."""
        agent = ValidationAgent(llm=llm)
        agent.interface = MagicMock(spec=pn.chat.ChatFeed)

        llm.set_responses([
            QueryCompletionValidation(
                chain_of_thought="Several aspects not addressed.",
                missing_elements=[
                    "Revenue by region",
                    "Year-over-year comparison",
                    "Top 5 products",
                ],
                suggestions=[
                    "Add GROUP BY region",
                    "Include LAG window function",
                    "Add ORDER BY revenue DESC LIMIT 5",
                ],
                correct=False,
            ),
        ])
        out, _ = await agent.respond(test_messages, {})
        assert len(out[0].missing_elements) == 3
        assert len(out[0].suggestions) == 3


# ---------------------------------------------------------------------------
# TableListAgent tests
# ---------------------------------------------------------------------------

class TestTableListAgent:

    def test_instantiation(self):
        """TableListAgent can be created with default params."""
        agent = TableListAgent()
        assert agent._column_name == "Data"
        assert agent._message_format is not None

    def test_conditions_mention_listing(self):
        """Conditions mention listing data."""
        agent = TableListAgent()
        conditions_text = " ".join(agent.conditions).lower()
        assert "list" in conditions_text

    def test_not_with_sql_and_dbtsl(self):
        """TableListAgent should not run alongside SQL/Dbtsl agents."""
        agent = TableListAgent()
        assert "SQLAgent" in agent.not_with
        assert "DbtslAgent" in agent.not_with

    async def test_applies_multiple_slugs(self):
        """applies() returns True when >1 visible slugs."""
        context = {"visible_slugs": {"src ⦙ table1", "src ⦙ table2"}}
        assert await TableListAgent.applies(context) is True

    async def test_applies_single_slug(self):
        """applies() returns False when only 1 visible slug."""
        context = {"visible_slugs": {"src ⦙ table1"}}
        assert await TableListAgent.applies(context) is False

    async def test_applies_empty_slugs(self):
        """applies() returns False when no visible slugs."""
        context = {"visible_slugs": set()}
        assert await TableListAgent.applies(context) is False

    async def test_applies_no_key(self):
        """applies() returns False when visible_slugs not in context."""
        assert await TableListAgent.applies({}) is False

    def test_get_items_groups_by_source(self):
        """_get_items groups tables by source name."""
        agent = TableListAgent()
        context = {
            "visible_slugs": {
                f"sales{SOURCE_TABLE_SEPARATOR}orders",
                f"sales{SOURCE_TABLE_SEPARATOR}customers",
                f"hr{SOURCE_TABLE_SEPARATOR}employees",
            }
        }
        items = agent._get_items(context)
        assert "sales" in items
        assert "hr" in items
        assert set(items["sales"]) == {"orders", "customers"}
        assert items["hr"] == ["employees"]

    def test_get_items_empty_slugs(self):
        """_get_items returns empty dict when no visible slugs."""
        agent = TableListAgent()
        assert agent._get_items({"visible_slugs": set()}) == {}

    def test_get_items_no_visible_slugs_key(self):
        """_get_items returns empty dict when key is missing."""
        agent = TableListAgent()
        assert agent._get_items({}) == {}

    def test_get_items_uses_closest_tables(self):
        """When closest_tables is in context, use it as search results."""
        agent = TableListAgent()
        context = {
            "closest_tables": ["orders", "customers"],
            "visible_slugs": {f"db{SOURCE_TABLE_SEPARATOR}orders"},
        }
        items = agent._get_items(context)
        assert "Search Results" in items
        assert items["Search Results"] == ["orders", "customers"]

    def test_get_items_with_metaset_ordering(self):
        """When metaset is available, items are ordered by catalog order."""
        agent = TableListAgent()
        slug1 = f"src{SOURCE_TABLE_SEPARATOR}alpha"
        slug2 = f"src{SOURCE_TABLE_SEPARATOR}beta"

        catalog = {
            slug2: TableCatalogEntry(
                table_slug=slug2, similarity=0.9,
                description="Beta table",
                columns=[Column(name="id")],
            ),
            slug1: TableCatalogEntry(
                table_slug=slug1, similarity=0.8,
                description="Alpha table",
                columns=[Column(name="id")],
            ),
        }
        metaset = Metaset(query="test", catalog=catalog)

        context = {
            "visible_slugs": {slug1, slug2},
            "metaset": metaset,
        }
        items = agent._get_items(context)
        assert items["src"] == ["beta", "alpha"]

    def test_create_row_content_with_metadata(self):
        """Row content function returns metadata markdown."""
        agent = TableListAgent()
        slug = f"mydb{SOURCE_TABLE_SEPARATOR}users"
        catalog = {
            slug: TableCatalogEntry(
                table_slug=slug, similarity=1.0,
                description="User accounts",
                columns=[Column(name="id"), Column(name="name")],
                metadata={"rows": 1000},
            ),
        }
        metaset = Metaset(query="test", catalog=catalog)
        context = {"metaset": metaset}

        row_fn = agent._create_row_content(context, "mydb")
        row = pd.Series({"Data": "users"})
        result = row_fn(row)
        assert isinstance(result, Markdown)
        assert "User accounts" in result.object
        assert "2 columns" in result.object
        assert "1000 rows" in result.object

    def test_create_row_content_no_metaset(self):
        """Row content returns 'No metadata available' when no metaset."""
        agent = TableListAgent()
        row_fn = agent._create_row_content({}, "src")

        row = pd.Series({"Data": "some_table"})
        result = row_fn(row)
        assert isinstance(result, Markdown)
        assert "No metadata" in result.object

    def test_create_row_content_missing_table_in_catalog(self):
        """Row content handles missing table slug gracefully."""
        agent = TableListAgent()
        existing_slug = f"src{SOURCE_TABLE_SEPARATOR}existing"
        catalog = {
            existing_slug: TableCatalogEntry(
                table_slug=existing_slug, similarity=1.0,
                description="exists",
                columns=[Column(name="id")],
            ),
        }
        metaset = Metaset(query="test", catalog=catalog)
        context = {"metaset": metaset}

        row_fn = agent._create_row_content(context, "src")
        row = pd.Series({"Data": "nonexistent"})
        result = row_fn(row)
        assert isinstance(result, Markdown)
        assert "No metadata" in result.object

    def test_create_row_content_no_description(self):
        """Row content works when catalog entry has no description."""
        agent = TableListAgent()
        slug = f"src{SOURCE_TABLE_SEPARATOR}data"
        catalog = {
            slug: TableCatalogEntry(
                table_slug=slug, similarity=1.0,
                description="",
                columns=[Column(name="x")],
            ),
        }
        metaset = Metaset(query="test", catalog=catalog)
        context = {"metaset": metaset}

        row_fn = agent._create_row_content(context, "src")
        row = pd.Series({"Data": "data"})
        result = row_fn(row)
        assert isinstance(result, Markdown)
        assert "1 columns" in result.object
