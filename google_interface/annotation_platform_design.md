# Google Apps Script Annotation Interface

This folder contains a minimal web-based annotation interface built on top of Google Apps Script, Google Drive, and Google Sheets. It was designed to annotate text transcripts (stored as `.txt` files) with a set of public speaking criteria and subjective ratings.

## Files

- `Code.gs` – server-side Apps Script:
  - Serves the HTML UI via `doGet()`.
  - Lists transcript files from a Google Drive folder (`FOLDER_ID`).
  - Loads the content of a selected transcript.
  - Saves annotation responses into a Google Sheet (`SHEET_NAME`).

- `Index.html` – client-side HTML/JS front-end:
  - Displays a drop-down list of available transcript IDs.
  - Shows the selected transcript text.
  - Presents the full annotation form (categorical criteria, Likert scales, global score slider, and free-text comment).
  - On submission, calls `saveAnnotation()` in `Code.gs` and appends a row to the Google Sheet.

## How it works

1. **Transcripts in Google Drive**

   - Create a folder in Google Drive and put your transcripts as `.txt` files.
   - Each file name should be an ID, e.g. `UCA01.txt`, `GRE13.txt`, etc.
   - Copy the folder ID from the URL and paste it into `FOLDER_ID` in `Code.gs`.

2. **Google Sheet for responses**

   - Create a Google Sheet and add a sheet named `Responses` (or change `SHEET_NAME` in `Code.gs`).
   - Add a header row matching the order in `saveAnnotation()` (ID, criteria A–C, scales, global score, comment, time).

3. **Apps Script setup**

   - In the Sheet, open **Extensions → Apps Script**.
   - Create a script project, add `Code.gs` and `Index.html` with the code from this repository.
   - Save and **Deploy → Test deployments** or **Deploy as web app**:
     - Execution: *Me*
     - Access: *Anyone with the link* (or more restricted, depending on your setup).

4. **Annotation workflow**

   - Open the web app URL.
   - Select a transcript ID from the drop-down (annotated files are marked as `🟢`, unannotated as `🔴`).
   - Load the transcript, read it, and fill out the form.
   - Click **Enregistrer l'Annotation**.
   - Each submission appends a new row to the `Responses` sheet, including the time spent on the annotation.

You can adapt the criteria texts, response options, and sheet columns to your own project, as long as the order of fields in `saveAnnotation()` matches the order of columns in your Google Sheet.
