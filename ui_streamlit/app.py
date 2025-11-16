"""Main Streamlit application."""

import streamlit as st
from pathlib import Path
from typing import List

from pipeline_runner import run_pipeline
from step_models import StepLog

OUTPUT_ROOT = Path("generated_data")  # relative to ui_streamlit/


def show_step_log(steps: List[StepLog]):
    """Display step logs in a nice, structured format."""
    label_map = {
        "manager": "Manager (Requirements)",
        "conceptual_designer": "Conceptual Designer",
        "logical_designer": "Logical Designer",
        "dist_engineer": "Distribution Engineer",
        "workload_designer": "Workload Designer",
        "qa_compiler": "QA & Compiler",
        "generation": "Data Generation",
    }
    for step in steps:
        label = label_map.get(step.name, step.name)
        if step.status == "done":
            icon = "✅"
        elif step.status == "running":
            icon = "⏳"
        elif step.status == "error":
            icon = "❌"
        else:
            icon = "⚪"
        st.markdown(f"**{icon} {label}** — {step.status.upper()}")
        if step.summary:
            st.caption(step.summary)
        if step.message and step.status == "error":
            st.error(step.message)


def main():
    """Main application entry point."""
    st.set_page_config(page_title="NL → Synthetic Data", layout="centered")

    st.title("NL → Synthetic Relational Data Generator")
    st.write(
        "Describe the dataset you want; the system will design schema, "
        "distributions, and generate CSVs."
    )

    # NL input
    example = (
        "Generate a retail sales dataset with one fact table and three dimension "
        "tables. The fact table should have 5 million rows. Product sales should "
        "follow a Zipf distribution. Customer purchases should be seasonal with "
        "peaks in December. The data should stress test group-by and join performance."
    )
    nl_text = st.text_area(
        "Dataset description",
        value=example,
        height=220,
    )

    run_button = st.button("Run pipeline", type="primary")

    # Initialize session state
    if "steps" not in st.session_state:
        st.session_state["steps"] = []
    if "tables" not in st.session_state:
        st.session_state["tables"] = []
    if "out_dir" not in st.session_state:
        st.session_state["out_dir"] = None
    if "ir" not in st.session_state:
        st.session_state["ir"] = None

    if run_button and nl_text.strip():
        # Clear previous results
        st.session_state["steps"] = []
        st.session_state["tables"] = []
        st.session_state["out_dir"] = None
        st.session_state["ir"] = None

        with st.spinner("Running multi-agent pipeline..."):
            try:
                ir, steps, out_dir, table_names = run_pipeline(
                    nl_description=nl_text,
                    output_root=OUTPUT_ROOT,
                )
                st.session_state["steps"] = steps
                st.session_state["tables"] = table_names
                st.session_state["out_dir"] = str(out_dir)
                st.session_state["ir"] = ir
                st.success("Pipeline completed successfully!")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback

                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    # Show process
    if st.session_state["steps"]:
        st.subheader("Pipeline steps")
        show_step_log(st.session_state["steps"])

    # Show schema summary if available
    if st.session_state["ir"]:
        st.subheader("Generated Schema")
        ir = st.session_state["ir"]
        for table_name, table in ir.logical.tables.items():
            with st.expander(f"📊 {table_name} ({table.kind or 'table'})"):
                st.write(f"**Row count:** {table.row_count or 'Not specified'}")
                st.write("**Columns:**")
                for col in table.columns:
                    col_info = f"- {col.name} ({col.sql_type})"
                    if col.role:
                        col_info += f" [{col.role}]"
                    if col.references:
                        col_info += f" → {col.references}"
                    st.write(col_info)
                if table.primary_key:
                    st.write(f"**Primary Key:** {', '.join(table.primary_key)}")
                if table.foreign_keys:
                    st.write("**Foreign Keys:**")
                    for fk in table.foreign_keys:
                        st.write(
                            f"- {fk.column} → {fk.ref_table}.{fk.ref_column}"
                        )

    # Show downloads once done
    if st.session_state["tables"] and st.session_state["out_dir"]:
        st.subheader("Download generated CSVs")
        out_dir = Path(st.session_state["out_dir"])
        for table in st.session_state["tables"]:
            csv_path = out_dir / f"{table}.csv"
            if csv_path.exists():
                with open(csv_path, "rb") as f:
                    file_size = csv_path.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    st.download_button(
                        label=f"Download {table}.csv ({size_mb:.2f} MB)",
                        data=f.read(),
                        file_name=f"{table}.csv",
                        mime="text/csv",
                        key=f"download-{table}",
                    )
            else:
                st.warning(f"File {table}.csv not found in {out_dir}")


if __name__ == "__main__":
    main()

