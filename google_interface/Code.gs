// Code.gs
// ---------------------------
// Configuration
// ---------------------------

// ID of the Google Drive folder containing .txt transcripts
const FOLDER_ID = 'REPLACE_WITH_YOUR_FOLDER_ID';

// Name of the sheet where annotation responses are stored
const SHEET_NAME = 'Responses';

// ---------------------------
// Web app entry point
// ---------------------------

function doGet() {
  // Serves the HTML front-end named "Index.html"
  return HtmlService.createHtmlOutputFromFile('Index');
}

// ---------------------------
// Helpers
// ---------------------------

function listFilesWithAnnotationStatus() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    throw new Error('Sheet "' + SHEET_NAME + '" not found. Please create it or update SHEET_NAME.');
  }

  const lastRow = sheet.getLastRow();
  let annotatedIds = new Set();

  // Collect IDs from the first column (starting row 2)
  if (lastRow > 1) {
    const idRange = sheet.getRange(2, 1, lastRow - 1, 1).getValues();
    annotatedIds = new Set(idRange.flat().filter(String));
  }

  const folder = DriveApp.getFolderById(FOLDER_ID);
  const files = folder.getFiles();
  const fileList = [];

  while (files.hasNext()) {
    const file = files.next();
    const fileId = file.getName().replace('.txt', '');
    const isAnnotated = annotatedIds.has(fileId);

    fileList.push({
      id: fileId,
      name: file.getName(),
      url: file.getUrl(),
      annotated: isAnnotated
    });
  }

  return fileList;
}

function getTranscriptContent(fileId) {
  const folder = DriveApp.getFolderById(FOLDER_ID);
  const files = folder.getFilesByName(fileId + '.txt');

  if (!files.hasNext()) {
    throw new Error('No .txt file found in folder for ID: ' + fileId);
  }

  const file = files.next();
  const content = file.getBlob().getDataAsString();
  Logger.log('Transcript content for file ID ' + fileId + ': ' + content);
  return content;
}

function saveAnnotation(fileId, responses, scaleResponses, globalScore, comment, timeElapsed) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    throw new Error('Sheet "' + SHEET_NAME + '" not found. Please create it or update SHEET_NAME.');
  }

  // Ensure data aligns with the column order in your Google Sheet
  const row = [
    fileId,              // ID
    responses[0],        // Topic Presentation
    responses[1],        // Structure
    responses[2],        // Language Level
    responses[3],        // Passive Voice
    responses[4],        // Conciseness
    responses[5],        // Redundancy
    responses[6],        // Negative Language
    responses[7],        // Metaphor
    responses[8],        // Storytelling
    scaleResponses[0],   // Introduction
    scaleResponses[1],   // Conclusion
    scaleResponses[2],   // Persuasiveness
    scaleResponses[3],   // Clarity of Language
    scaleResponses[4],   // Creativity of Speech
    globalScore,         // Global Score (1–100)
    comment,             // Comment (optional)
    timeElapsed          // Time elapsed (seconds)
  ];

  sheet.appendRow(row);
  return 'Annotation saved.';
}
