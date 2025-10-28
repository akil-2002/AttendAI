# AttendAI

## Job Application Tracker CLI

This repository now includes a small command-line helper for keeping track of job applications.

### Usage

Run the tool with Python:

```bash
python job_tracker.py list
```

#### Add a new application

```bash
python job_tracker.py add "Acme Corp" "Software Engineer" --status "Applied" --notes "Referral from Alex"
```

#### Update an existing entry

Use the index from the `list` command (1-based):

```bash
python job_tracker.py update 1 --status "Phone Interview"
```

#### Delete an application

```bash
python job_tracker.py delete 1
```

All entries are stored locally in `job_applications.json` alongside the script.