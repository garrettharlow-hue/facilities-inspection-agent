import json
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from datetime import datetime
from openai import OpenAI


# =========================================================
# CONFIGURATION
# =========================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

MODEL_NAME = os.environ.get(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)

csv_path = "facilities_inspections.csv"
work_order_csv_path = "facilities_work_orders.csv"


# Seed the deployed demo with synthetic sample data
if (
    not os.path.exists(csv_path)
    and os.path.exists("sample_data/sample_inspections.csv")
):
    shutil.copy(
        "sample_data/sample_inspections.csv",
        csv_path
    )

if (
    not os.path.exists(work_order_csv_path)
    and os.path.exists("sample_data/sample_work_orders.csv")
):
    shutil.copy(
        "sample_data/sample_work_orders.csv",
        work_order_csv_path
    )


# =========================================================
# STANDARDIZED VALUES
# =========================================================

valid_occupancy = [
    "Vacant",
    "Occupied",
    "Unknown"
]

valid_carpet = [
    "Good",
    "Stained",
    "Damaged",
    "Unknown"
]

valid_furniture = [
    "Good",
    "Minor Issue",
    "Damaged",
    "Missing",
    "Unknown"
]

valid_cleanliness = [
    "Clean",
    "Acceptable",
    "Needs Cleaning",
    "Unknown"
]

valid_safety = [
    "None",
    "Issue Identified",
    "Unknown"
]

required_inspection_fields = [
    "building",
    "room",
    "occupancy_status",
    "carpet_condition",
    "furniture_condition",
    "cleanliness",
    "safety_issue"
]

valid_work_order_statuses = [
    "Open",
    "In Progress",
    "Completed"
]


# =========================================================
# VALIDATION AND BUSINESS RULES
# =========================================================

def validate_record(record):
    errors = []

    if record["occupancy_status"] not in valid_occupancy:
        errors.append("Invalid occupancy status")

    if record["carpet_condition"] not in valid_carpet:
        errors.append("Invalid carpet condition")

    if record["furniture_condition"] not in valid_furniture:
        errors.append("Invalid furniture condition")

    if record["cleanliness"] not in valid_cleanliness:
        errors.append("Invalid cleanliness value")

    if record["safety_issue"] not in valid_safety:
        errors.append("Invalid safety value")

    return errors


def determine_maintenance(record):
    if record["carpet_condition"] in [
        "Stained",
        "Damaged"
    ]:
        return True

    if record["furniture_condition"] in [
        "Minor Issue",
        "Damaged",
        "Missing"
    ]:
        return True

    if record["cleanliness"] == "Needs Cleaning":
        return True

    if record["safety_issue"] == "Issue Identified":
        return True

    return False


def determine_priority(record):
    if record["safety_issue"] == "Issue Identified":
        return "High"

    if record["carpet_condition"] == "Damaged":
        return "Medium"

    if record["furniture_condition"] in [
        "Damaged",
        "Missing"
    ]:
        return "Medium"

    if record["carpet_condition"] == "Stained":
        return "Low"

    if record["furniture_condition"] == "Minor Issue":
        return "Low"

    if record["cleanliness"] == "Needs Cleaning":
        return "Low"

    return "None"


def check_missing_fields(record):
    missing_fields = []

    for field in required_inspection_fields:
        value = record.get(field)

        if (
            value is None
            or str(value).strip() == ""
            or str(value).strip() == "Unknown"
        ):
            missing_fields.append(field)

    return missing_fields


field_labels = {
    "building": "building",
    "room": "room number",
    "occupancy_status": "occupancy status",
    "carpet_condition": "carpet condition",
    "furniture_condition": "furniture condition",
    "cleanliness": "cleanliness",
    "safety_issue": "safety condition"
}


def generate_follow_up_question(missing_fields):
    if len(missing_fields) == 0:
        return "No additional information is needed."

    readable_fields = [
        field_labels[field]
        for field in missing_fields
    ]

    if len(readable_fields) == 1:
        field_text = readable_fields[0]

    elif len(readable_fields) == 2:
        field_text = " and ".join(readable_fields)

    else:
        field_text = (
            ", ".join(readable_fields[:-1])
            + ", and "
            + readable_fields[-1]
        )

    return f"Please provide the {field_text}."


def determine_next_action(record):
    missing_fields = check_missing_fields(record)

    if len(missing_fields) > 0:
        return {
            "status": "Needs Follow-Up",
            "missing_fields": missing_fields,
            "message": generate_follow_up_question(
                missing_fields
            )
        }

    return {
        "status": "Ready to Save",
        "missing_fields": [],
        "message": (
            "Inspection record is complete and ready to save."
        )
    }


def create_standardized_summary(record):
    return (
        f"{record['building']} {record['room']} | "
        f"Occupancy: {record['occupancy_status']} | "
        f"Carpet: {record['carpet_condition']} | "
        f"Furniture: {record['furniture_condition']} | "
        f"Cleanliness: {record['cleanliness']} | "
        f"Safety: {record['safety_issue']}."
    )


# =========================================================
# LLM PROCESSING
# =========================================================

def parse_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if (
            len(lines) > 0
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(lines)

    return json.loads(text)


def process_inspection(note):
    response = client.responses.create(
        model=MODEL_NAME,
        instructions="""
You are a facilities inspection documentation assistant.

Convert the inspector's natural-language notes into standardized inspection data.

Rules:
- Use only information explicitly provided by the inspector.
- Do not assume something is in good condition if it was not mentioned.
- If a condition was not observed or mentioned, record it as "Unknown".
- If the building or room is not provided, record it as "Unknown".
- If a safety item is explicitly observed and described as normal, functional, okay, or having no concern, classify Safety Issue as "None".
- Use "Unknown" for Safety Issue only when safety conditions were not observed or not mentioned.
- Use only the allowed standardized values listed below.

Allowed values:

Occupancy Status:
- Vacant
- Occupied
- Unknown

Carpet Condition:
- Good
- Stained
- Damaged
- Unknown

Furniture Condition:
- Good
- Minor Issue
- Damaged
- Missing
- Unknown

Cleanliness:
- Clean
- Acceptable
- Needs Cleaning
- Unknown

Safety Issue:
- None
- Issue Identified
- Unknown

Return ONLY valid JSON.
""",
        input=f"""
Inspection note:

{note}

Return JSON with exactly these fields:

building
room
occupancy_status
carpet_condition
furniture_condition
cleanliness
safety_issue
summary
"""
    )

    record = parse_json_response(
        response.output_text
    )

    for field in required_inspection_fields:
        if field not in record:
            record[field] = "Unknown"

    record["original_notes"] = note

    record["maintenance_required"] = (
        determine_maintenance(record)
    )

    record["priority"] = (
        determine_priority(record)
    )

    if not record.get("summary"):
        record["summary"] = (
            create_standardized_summary(record)
        )

    return record


def update_record_from_followup(
    record,
    followup_text
):
    missing_fields = check_missing_fields(record)

    if len(missing_fields) == 0:
        return record.copy()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions="""
You are updating a facilities inspection record using additional information provided by an inspector.

Rules:
- Update only fields that were previously marked as missing or "Unknown".
- Do not change information that was already recorded.
- Use only information explicitly provided in the follow-up response.
- If the follow-up still does not provide enough information for a field, keep it as "Unknown".
- If a safety item is explicitly described as normal, functional, okay, or having no concern, classify Safety Issue as "None".

Allowed values:

Occupancy Status:
- Vacant
- Occupied
- Unknown

Carpet Condition:
- Good
- Stained
- Damaged
- Unknown

Furniture Condition:
- Good
- Minor Issue
- Damaged
- Missing
- Unknown

Cleanliness:
- Clean
- Acceptable
- Needs Cleaning
- Unknown

Safety Issue:
- None
- Issue Identified
- Unknown

Return ONLY valid JSON.
""",
        input=f"""
Fields that are currently missing:

{missing_fields}

Inspector follow-up response:

{followup_text}

Return JSON containing only the missing fields.
"""
    )

    updates = parse_json_response(
        response.output_text
    )

    updated_record = record.copy()

    for field in missing_fields:
        if field in updates:
            updated_record[field] = updates[field]

    existing_notes = updated_record.get(
        "original_notes",
        ""
    )

    updated_record["original_notes"] = (
        existing_notes
        + " Follow-up: "
        + followup_text
    ).strip()

    updated_record["maintenance_required"] = (
        determine_maintenance(updated_record)
    )

    updated_record["priority"] = (
        determine_priority(updated_record)
    )

    updated_record["summary"] = (
        create_standardized_summary(
            updated_record
        )
    )

    return updated_record


# =========================================================
# HUMAN EDITING
# =========================================================

def apply_manual_edits(
    record,
    building,
    room,
    occupancy_status,
    carpet_condition,
    furniture_condition,
    cleanliness,
    safety_issue
):
    if record is None:
        return None

    updated_record = record.copy()

    new_values = {
        "building": building,
        "room": room,
        "occupancy_status": occupancy_status,
        "carpet_condition": carpet_condition,
        "furniture_condition": furniture_condition,
        "cleanliness": cleanliness,
        "safety_issue": safety_issue
    }

    changes = []

    for field, new_value in new_values.items():
        old_value = updated_record.get(field)

        if str(old_value) != str(new_value):
            changes.append(
                f"{field}: {old_value} -> {new_value}"
            )

        updated_record[field] = new_value

    if len(changes) > 0:
        existing_notes = updated_record.get(
            "original_notes",
            ""
        )

        updated_record["original_notes"] = (
            existing_notes
            + " Manual edits: "
            + "; ".join(changes)
        ).strip()

    updated_record["maintenance_required"] = (
        determine_maintenance(updated_record)
    )

    updated_record["priority"] = (
        determine_priority(updated_record)
    )

    updated_record["summary"] = (
        create_standardized_summary(
            updated_record
        )
    )

    return updated_record


def populate_review_fields(record):
    if record is None:
        return (
            "",
            "",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown"
        )

    return (
        record.get("building", ""),
        record.get("room", ""),
        record.get(
            "occupancy_status",
            "Unknown"
        ),
        record.get(
            "carpet_condition",
            "Unknown"
        ),
        record.get(
            "furniture_condition",
            "Unknown"
        ),
        record.get(
            "cleanliness",
            "Unknown"
        ),
        record.get(
            "safety_issue",
            "Unknown"
        )
    )


# =========================================================
# INSPECTION STORAGE
# =========================================================

INSPECTION_COLUMNS = [
    "building",
    "room",
    "occupancy_status",
    "carpet_condition",
    "furniture_condition",
    "cleanliness",
    "safety_issue",
    "summary",
    "original_notes",
    "maintenance_required",
    "priority"
]


def load_inspections():
    if not os.path.exists(csv_path):
        return pd.DataFrame(
            columns=INSPECTION_COLUMNS
        )

    df = pd.read_csv(
        csv_path,
        keep_default_na=False
    )

    unwanted_columns = [
        column
        for column in df.columns
        if (
            column.startswith("Unnamed")
            or column == "index"
        )
    ]

    if len(unwanted_columns) > 0:
        df = df.drop(
            columns=unwanted_columns
        )

    for column in INSPECTION_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[INSPECTION_COLUMNS]


def inspection_exists(building, room):
    df = load_inspections()

    if df.empty:
        return False

    duplicate = (
        (
            df["building"].astype(str)
            == str(building)
        )
        &
        (
            df["room"].astype(str)
            == str(room)
        )
    )

    return duplicate.any()


def save_inspection(record):
    action = determine_next_action(record)

    if action["status"] != "Ready to Save":
        return {
            "saved": False,
            "message": action["message"]
        }

    errors = validate_record(record)

    if len(errors) > 0:
        return {
            "saved": False,
            "message": (
                "Record contains validation errors."
            ),
            "errors": errors
        }

    if inspection_exists(
        record["building"],
        record["room"]
    ):
        return {
            "saved": False,
            "message": (
                "An inspection record already exists "
                "for this building and room."
            ),
            "building": record["building"],
            "room": record["room"]
        }

    existing_df = load_inspections()

    new_row = {
        column: record.get(column, "")
        for column in INSPECTION_COLUMNS
    }

    new_df = pd.DataFrame(
        [new_row]
    )

    updated_df = pd.concat(
        [
            existing_df,
            new_df
        ],
        ignore_index=True
    )

    updated_df.to_csv(
        csv_path,
        index=False
    )

    return {
        "saved": True,
        "message": (
            "Inspection record saved successfully."
        ),
        "building": record["building"],
        "room": record["room"]
    }


# =========================================================
# WORKFLOW CONTROLLER
# =========================================================

def start_inspection_for_review(note):
    record = process_inspection(note)

    action = determine_next_action(record)

    if action["status"] == "Needs Follow-Up":
        return {
            "status": "Needs Follow-Up",
            "record": record,
            "missing_fields": (
                action["missing_fields"]
            ),
            "message": action["message"]
        }

    return {
        "status": "Ready for Review",
        "record": record,
        "message": (
            "Inspection is complete. "
            "Please review the record before saving."
        )
    }


def continue_inspection_for_review(
    record,
    followup_text
):
    updated_record = (
        update_record_from_followup(
            record,
            followup_text
        )
    )

    action = determine_next_action(
        updated_record
    )

    if action["status"] == "Needs Follow-Up":
        return {
            "status": "Needs Follow-Up",
            "record": updated_record,
            "missing_fields": (
                action["missing_fields"]
            ),
            "message": action["message"]
        }

    return {
        "status": "Ready for Review",
        "record": updated_record,
        "message": (
            "Inspection is complete. "
            "Please review the record before saving."
        )
    }


# =========================================================
# MAINTENANCE WORK ORDERS
# =========================================================

WORK_ORDER_COLUMNS = [
    "work_order_id",
    "building",
    "room",
    "category",
    "priority",
    "issue_description",
    "source",
    "status",
    "created_at",
    "updated_at",
    "completed_at"
]


def load_work_orders():
    if not os.path.exists(
        work_order_csv_path
    ):
        return pd.DataFrame(
            columns=WORK_ORDER_COLUMNS
        )

    df = pd.read_csv(
        work_order_csv_path,
        keep_default_na=False
    )

    unwanted_columns = [
        column
        for column in df.columns
        if (
            column.startswith("Unnamed")
            or column == "index"
        )
    ]

    if len(unwanted_columns) > 0:
        df = df.drop(
            columns=unwanted_columns
        )

    for column in WORK_ORDER_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[WORK_ORDER_COLUMNS]


def identify_work_order_categories(record):
    categories = []

    if record["carpet_condition"] in [
        "Stained",
        "Damaged"
    ]:
        categories.append("Carpet")

    if record["furniture_condition"] in [
        "Minor Issue",
        "Damaged",
        "Missing"
    ]:
        categories.append("Furniture")

    if record["cleanliness"] == "Needs Cleaning":
        categories.append("Cleaning")

    if record["safety_issue"] == "Issue Identified":
        categories.append("Safety")

    return categories


def generate_work_order(record):
    maintenance_required = (
        str(
            record["maintenance_required"]
        ).lower()
        == "true"
    )

    if not maintenance_required:
        return None

    categories = (
        identify_work_order_categories(
            record
        )
    )

    current_time = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    return {
        "work_order_id": (
            f"WO-{record['building']}-{record['room']}"
        ),
        "building": record["building"],
        "room": record["room"],
        "category": ", ".join(categories),
        "priority": record["priority"],
        "issue_description": (
            record["summary"]
        ),
        "source": (
            "Facilities Inspection Agent"
        ),
        "status": "Open",
        "created_at": current_time,
        "updated_at": "",
        "completed_at": ""
    }


def save_work_order(work_order):
    if work_order is None:
        return {
            "created": False,
            "message": (
                "No maintenance work order is required."
            )
        }

    existing_df = load_work_orders()

    if not existing_df.empty:
        duplicate = (
            existing_df[
                "work_order_id"
            ].astype(str)
            == str(
                work_order[
                    "work_order_id"
                ]
            )
        )

        if duplicate.any():
            return {
                "created": False,
                "message": (
                    "A work order already exists "
                    "for this inspection."
                )
            }

    new_df = pd.DataFrame(
        [work_order]
    )

    updated_df = pd.concat(
        [
            existing_df,
            new_df
        ],
        ignore_index=True
    )

    updated_df.to_csv(
        work_order_csv_path,
        index=False
    )

    return {
        "created": True,
        "message": (
            "Maintenance work order "
            "created successfully."
        ),
        "work_order": work_order
    }


def get_work_order_history():
    df = load_work_orders()

    display_columns = [
        "Work Order ID",
        "Building",
        "Room",
        "Category",
        "Priority",
        "Issue Description",
        "Status",
        "Created At",
        "Updated At",
        "Completed At"
    ]

    if df.empty:
        return pd.DataFrame(
            columns=display_columns
        )

    history_df = df.copy()

    history_df.columns = [
        "Work Order ID",
        "Building",
        "Room",
        "Category",
        "Priority",
        "Issue Description",
        "Source",
        "Status",
        "Created At",
        "Updated At",
        "Completed At"
    ]

    return history_df[
        display_columns
    ]


def get_work_order_ids():
    df = load_work_orders()

    if df.empty:
        return []

    return (
        df["work_order_id"]
        .astype(str)
        .tolist()
    )


def update_work_order_status(
    work_order_id,
    new_status
):
    if new_status not in valid_work_order_statuses:
        return {
            "updated": False,
            "message": (
                "Invalid work-order status."
            )
        }

    df = load_work_orders()

    if df.empty:
        return {
            "updated": False,
            "message": (
                "No maintenance work orders were found."
            )
        }

    matching_rows = (
        df["work_order_id"]
        .astype(str)
        == str(work_order_id)
    )

    if not matching_rows.any():
        return {
            "updated": False,
            "message": (
                "Work order was not found."
            )
        }

    current_time = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    df.loc[
        matching_rows,
        "status"
    ] = new_status

    df.loc[
        matching_rows,
        "updated_at"
    ] = current_time

    if new_status == "Completed":
        df.loc[
            matching_rows,
            "completed_at"
        ] = current_time
    else:
        df.loc[
            matching_rows,
            "completed_at"
        ] = ""

    df.to_csv(
        work_order_csv_path,
        index=False
    )

    return {
        "updated": True,
        "message": (
            f"{work_order_id} "
            f"updated to {new_status}."
        )
    }


# =========================================================
# ANALYTICS
# =========================================================

def get_building_summary():
    df = load_inspections()

    if df.empty:
        return pd.DataFrame(
            [
                [
                    "Status",
                    "No inspection records found."
                ]
            ],
            columns=[
                "Metric",
                "Value"
            ]
        )

    total_rooms = len(df)

    maintenance_rooms = (
        df["maintenance_required"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    no_maintenance_rooms = (
        total_rooms
        - maintenance_rooms
    )

    priority_counts = (
        df["priority"]
        .value_counts()
        .reindex(
            [
                "High",
                "Medium",
                "Low",
                "None"
            ],
            fill_value=0
        )
    )

    carpet_issues = df[
        df["carpet_condition"].isin(
            [
                "Stained",
                "Damaged"
            ]
        )
    ].shape[0]

    furniture_issues = df[
        df["furniture_condition"].isin(
            [
                "Minor Issue",
                "Damaged",
                "Missing"
            ]
        )
    ].shape[0]

    cleaning_issues = df[
        df["cleanliness"]
        == "Needs Cleaning"
    ].shape[0]

    safety_issues = df[
        df["safety_issue"]
        == "Issue Identified"
    ].shape[0]

    building_name = ", ".join(
        sorted(
            df["building"]
            .astype(str)
            .unique()
        )
    )

    summary = {
        "Building": building_name,
        "Rooms Inspected": total_rooms,
        "Maintenance Required": int(
            maintenance_rooms
        ),
        "No Maintenance Required": int(
            no_maintenance_rooms
        ),
        "High Priority": int(
            priority_counts["High"]
        ),
        "Medium Priority": int(
            priority_counts["Medium"]
        ),
        "Low Priority": int(
            priority_counts["Low"]
        ),
        "Carpet Issues": int(
            carpet_issues
        ),
        "Furniture Issues": int(
            furniture_issues
        ),
        "Cleaning Issues": int(
            cleaning_issues
        ),
        "Safety Issues": int(
            safety_issues
        )
    }

    return pd.DataFrame(
        list(summary.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )


def create_priority_chart():
    df = load_inspections()

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No inspection records available.",
            ha="center",
            va="center"
        )

        ax.axis("off")
        fig.tight_layout()
        return fig

    priority_counts = (
        df["priority"]
        .value_counts()
        .reindex(
            [
                "High",
                "Medium",
                "Low",
                "None"
            ],
            fill_value=0
        )
    )

    ax.bar(
        priority_counts.index,
        priority_counts.values
    )

    ax.set_title(
        "Maintenance Priority Distribution"
    )

    ax.set_xlabel(
        "Priority Level"
    )

    ax.set_ylabel(
        "Number of Rooms"
    )

    maximum = max(
        priority_counts.values
    )

    ax.set_ylim(
        0,
        maximum + 1
    )

    for index, value in enumerate(
        priority_counts.values
    ):
        ax.text(
            index,
            value + 0.05,
            str(value),
            ha="center"
        )

    fig.tight_layout()

    return fig


def create_issue_chart():
    df = load_inspections()

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No inspection records available.",
            ha="center",
            va="center"
        )

        ax.axis("off")
        fig.tight_layout()
        return fig

    issue_counts = {
        "Carpet": df[
            df["carpet_condition"].isin(
                [
                    "Stained",
                    "Damaged"
                ]
            )
        ].shape[0],

        "Furniture": df[
            df["furniture_condition"].isin(
                [
                    "Minor Issue",
                    "Damaged",
                    "Missing"
                ]
            )
        ].shape[0],

        "Cleaning": df[
            df["cleanliness"]
            == "Needs Cleaning"
        ].shape[0],

        "Safety": df[
            df["safety_issue"]
            == "Issue Identified"
        ].shape[0]
    }

    ax.bar(
        issue_counts.keys(),
        issue_counts.values()
    )

    ax.set_title(
        "Inspection Issues by Category"
    )

    ax.set_xlabel(
        "Issue Category"
    )

    ax.set_ylabel(
        "Number of Rooms"
    )

    maximum = max(
        issue_counts.values()
    )

    ax.set_ylim(
        0,
        maximum + 1
    )

    for index, value in enumerate(
        issue_counts.values()
    ):
        ax.text(
            index,
            value + 0.05,
            str(value),
            ha="center"
        )

    fig.tight_layout()

    return fig


def get_inspection_history():
    df = load_inspections()

    display_columns = [
        "Building",
        "Room",
        "Occupancy",
        "Carpet",
        "Furniture",
        "Cleanliness",
        "Safety",
        "Maintenance Required",
        "Priority"
    ]

    if df.empty:
        return pd.DataFrame(
            columns=display_columns
        )

    history_df = df[
        [
            "building",
            "room",
            "occupancy_status",
            "carpet_condition",
            "furniture_condition",
            "cleanliness",
            "safety_issue",
            "maintenance_required",
            "priority"
        ]
    ].copy()

    history_df.columns = (
        display_columns
    )

    history_df["_room_sort"] = (
        pd.to_numeric(
            history_df["Room"],
            errors="coerce"
        )
    )

    history_df = (
        history_df
        .sort_values(
            by=[
                "Building",
                "_room_sort",
                "Room"
            ]
        )
        .drop(
            columns=["_room_sort"]
        )
    )

    return history_df


# =========================================================
# UI FUNCTIONS
# =========================================================

def format_record(record):
    if record is None:
        return {}

    return {
        "Building": record.get("building"),
        "Room": record.get("room"),
        "Occupancy Status": record.get(
            "occupancy_status"
        ),
        "Carpet Condition": record.get(
            "carpet_condition"
        ),
        "Furniture Condition": record.get(
            "furniture_condition"
        ),
        "Cleanliness": record.get(
            "cleanliness"
        ),
        "Safety Issue": record.get(
            "safety_issue"
        ),
        "Maintenance Required": record.get(
            "maintenance_required"
        ),
        "Priority": record.get(
            "priority"
        ),
        "Summary": record.get(
            "summary"
        )
    }


def ui_apply_manual_edits(
    record,
    building,
    room,
    occupancy_status,
    carpet_condition,
    furniture_condition,
    cleanliness,
    safety_issue
):
    if record is None:
        return (
            "No inspection record is available to edit.",
            {},
            record
        )

    updated_record = apply_manual_edits(
        record,
        building,
        room,
        occupancy_status,
        carpet_condition,
        furniture_condition,
        cleanliness,
        safety_issue
    )

    errors = validate_record(
        updated_record
    )

    if len(errors) > 0:
        return (
            (
                "The edited record contains "
                "validation errors."
            ),
            format_record(
                updated_record
            ),
            updated_record
        )

    missing_fields = (
        check_missing_fields(
            updated_record
        )
    )

    if len(missing_fields) > 0:
        return (
            (
                "The edited record still contains "
                "missing required information."
            ),
            format_record(
                updated_record
            ),
            updated_record
        )

    return (
        (
            "Edits applied successfully. "
            "Review the updated record and confirm when ready."
        ),
        format_record(
            updated_record
        ),
        updated_record
    )


def ui_start_inspection_review(note):
    if not note.strip():
        return (
            "Please enter an inspection note.",
            {},
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    result = (
        start_inspection_for_review(
            note
        )
    )

    record = result["record"]

    if result["status"] == "Needs Follow-Up":
        return (
            (
                "Additional information is required.\n\n"
                + result["message"]
            ),
            format_record(record),
            record,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    return (
        (
            "Inspection complete. "
            "Please review the structured record before saving."
        ),
        format_record(record),
        record,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True)
    )


def ui_continue_inspection_review(
    record,
    followup_text
):
    if record is None:
        return (
            "No active inspection was found.",
            {},
            record,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    if not followup_text.strip():
        return (
            (
                "Please enter the requested "
                "follow-up information."
            ),
            format_record(record),
            record,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    result = (
        continue_inspection_for_review(
            record,
            followup_text
        )
    )

    updated_record = result["record"]

    if result["status"] == "Needs Follow-Up":
        return (
            (
                "Additional information is still required.\n\n"
                + result["message"]
            ),
            format_record(
                updated_record
            ),
            updated_record,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    return (
        (
            "Inspection complete. "
            "Please review the structured record before saving."
        ),
        format_record(
            updated_record
        ),
        updated_record,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True)
    )


def ui_confirm_save(record):
    if record is None:
        return (
            (
                "No inspection record is "
                "available to save."
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                value={},
                visible=False
            )
        )

    save_result = save_inspection(record)

    if not save_result["saved"]:
        return (
            save_result["message"],
            gr.update(
                visible=True
            ),
            gr.update(
                value={},
                visible=False
            )
        )

    work_order = generate_work_order(
        record
    )

    if work_order is not None:
        work_order_result = (
            save_work_order(
                work_order
            )
        )

        if work_order_result["created"]:
            return (
                (
                    "Inspection confirmed and saved successfully. "
                    "A maintenance work order was also created."
                ),
                gr.update(
                    visible=False
                ),
                gr.update(
                    value=work_order,
                    visible=True
                )
            )

        return (
            (
                "Inspection saved successfully. "
                + work_order_result["message"]
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                value=work_order,
                visible=True
            )
        )

    return (
        (
            "Inspection confirmed and saved successfully. "
            "No maintenance work order was required."
        ),
        gr.update(
            visible=False
        ),
        gr.update(
            value={},
            visible=False
        )
    )


def ui_show_building_dashboard():
    summary_df = get_building_summary()

    priority_fig = create_priority_chart()
    issue_fig = create_issue_chart()

    return (
        gr.update(
            value=summary_df,
            visible=True
        ),
        gr.update(
            value=priority_fig,
            visible=True
        ),
        gr.update(
            value=issue_fig,
            visible=True
        )
    )


def ui_show_inspection_history():
    history_df = (
        get_inspection_history()
    )

    return gr.update(
        value=history_df,
        visible=True
    )


def ui_refresh_work_orders():
    history_df = (
        get_work_order_history()
    )

    work_order_ids = (
        get_work_order_ids()
    )

    if len(work_order_ids) > 0:
        selected_id = work_order_ids[0]
    else:
        selected_id = None

    return (
        gr.update(
            value=history_df,
            visible=True
        ),
        gr.update(
            choices=work_order_ids,
            value=selected_id
        )
    )


def ui_update_work_order_status(
    work_order_id,
    new_status
):
    if not work_order_id:
        return (
            "Please select a work order.",
            get_work_order_history()
        )

    result = (
        update_work_order_status(
            work_order_id,
            new_status
        )
    )

    return (
        result["message"],
        get_work_order_history()
    )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Facilities Inspection Agent"
) as demo:

    gr.Markdown(
        """
        # Facilities Inspection Agent

        Enter inspection observations in natural language.
        The agent will standardize the information, identify missing
        details, and present the completed record for human review
        before saving.

        **Portfolio Demo:** Please use synthetic or test information only.
        """
    )

    current_record = gr.State(
        value=None
    )

    # NEW INSPECTION
    gr.Markdown(
        "## New Inspection"
    )

    inspection_note = gr.Textbox(
        label="Inspection Notes",
        placeholder=(
            "Example: Demo Hall 105 vacant. "
            "Carpet is stained near the doorway."
        ),
        lines=4
    )

    process_button = gr.Button(
        "Process Inspection"
    )

    status_output = gr.Textbox(
        label="Agent Status",
        interactive=False,
        lines=3
    )

    record_output = gr.JSON(
        label="Structured Inspection Record"
    )

    # HUMAN REVIEW
    gr.Markdown(
        "## Review and Edit Inspection Record"
    )

    with gr.Row():
        edit_building = gr.Textbox(
            label="Building"
        )

        edit_room = gr.Textbox(
            label="Room"
        )

    edit_occupancy = gr.Dropdown(
        choices=valid_occupancy,
        value="Unknown",
        label="Occupancy Status"
    )

    edit_carpet = gr.Dropdown(
        choices=valid_carpet,
        value="Unknown",
        label="Carpet Condition"
    )

    edit_furniture = gr.Dropdown(
        choices=valid_furniture,
        value="Unknown",
        label="Furniture Condition"
    )

    edit_cleanliness = gr.Dropdown(
        choices=valid_cleanliness,
        value="Unknown",
        label="Cleanliness"
    )

    edit_safety = gr.Dropdown(
        choices=valid_safety,
        value="Unknown",
        label="Safety Issue"
    )

    apply_edits_button = gr.Button(
        "Apply Edits"
    )

    # FOLLOW-UP
    followup_text = gr.Textbox(
        label="Follow-Up Information",
        placeholder=(
            "Example: Furniture looks good. "
            "Room is clean. No safety concerns."
        ),
        lines=3,
        visible=False
    )

    followup_button = gr.Button(
        "Submit Follow-Up",
        visible=False
    )

    # CONFIRMATION
    confirm_button = gr.Button(
        "Confirm & Save Inspection",
        visible=False
    )

    work_order_output = gr.JSON(
        label="Generated Maintenance Work Order",
        visible=False
    )

    # ANALYTICS
    gr.Markdown(
        "## Building Analytics"
    )

    summary_button = gr.Button(
        "View Building Summary"
    )

    summary_output = gr.Dataframe(
        headers=[
            "Metric",
            "Value"
        ],
        label="Building Inspection Summary",
        interactive=False,
        visible=False
    )

    priority_plot = gr.Plot(
        label="Maintenance Priority Distribution",
        visible=False
    )

    issue_plot = gr.Plot(
        label="Inspection Issues by Category",
        visible=False
    )

    # INSPECTION HISTORY
    gr.Markdown(
        "## Inspection History"
    )

    history_button = gr.Button(
        "View Inspection History"
    )

    history_output = gr.Dataframe(
        label="Saved Inspection Records",
        interactive=False,
        visible=False
    )

    # WORK ORDERS
    gr.Markdown(
        "## Maintenance Work Orders"
    )

    refresh_work_orders_button = gr.Button(
        "View / Refresh Work Orders"
    )

    work_order_history_output = gr.Dataframe(
        label="Maintenance Work Orders",
        interactive=False,
        visible=False
    )

    work_order_selector = gr.Dropdown(
        choices=get_work_order_ids(),
        label="Select Work Order"
    )

    work_order_status_selector = gr.Dropdown(
        choices=valid_work_order_statuses,
        value="Open",
        label="New Status"
    )

    update_work_order_button = gr.Button(
        "Update Work Order Status"
    )

    work_order_status_message = gr.Textbox(
        label="Work Order Update",
        interactive=False
    )

    # =====================================================
    # EVENTS
    # =====================================================

    process_button.click(
        fn=ui_start_inspection_review,
        inputs=inspection_note,
        outputs=[
            status_output,
            record_output,
            current_record,
            followup_text,
            followup_button,
            confirm_button
        ]
    )

    followup_button.click(
        fn=ui_continue_inspection_review,
        inputs=[
            current_record,
            followup_text
        ],
        outputs=[
            status_output,
            record_output,
            current_record,
            followup_text,
            followup_button,
            confirm_button
        ]
    )

    current_record.change(
        fn=populate_review_fields,
        inputs=current_record,
        outputs=[
            edit_building,
            edit_room,
            edit_occupancy,
            edit_carpet,
            edit_furniture,
            edit_cleanliness,
            edit_safety
        ]
    )

    apply_edits_button.click(
        fn=ui_apply_manual_edits,
        inputs=[
            current_record,
            edit_building,
            edit_room,
            edit_occupancy,
            edit_carpet,
            edit_furniture,
            edit_cleanliness,
            edit_safety
        ],
        outputs=[
            status_output,
            record_output,
            current_record
        ]
    )

    confirm_button.click(
        fn=ui_confirm_save,
        inputs=current_record,
        outputs=[
            status_output,
            confirm_button,
            work_order_output
        ]
    )

    summary_button.click(
        fn=ui_show_building_dashboard,
        inputs=[],
        outputs=[
            summary_output,
            priority_plot,
            issue_plot
        ]
    )

    history_button.click(
        fn=ui_show_inspection_history,
        inputs=[],
        outputs=history_output
    )

    refresh_work_orders_button.click(
        fn=ui_refresh_work_orders,
        inputs=[],
        outputs=[
            work_order_history_output,
            work_order_selector
        ]
    )

    update_work_order_button.click(
        fn=ui_update_work_order_status,
        inputs=[
            work_order_selector,
            work_order_status_selector
        ],
        outputs=[
            work_order_status_message,
            work_order_history_output
        ]
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
