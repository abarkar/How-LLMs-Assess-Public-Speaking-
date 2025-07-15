#!/bin/bash

# New variables
dimension="persuasiveness"
criteria="metaphor"
version="1"

# Create a directory for prompts if it doesn't exist
mkdir -p prompts

# Read content from task.txt, removing any carriage return characters
# task_content=$(tr -d '\r' < ./criteria_task/$dimension/$version/$criteria.txt | tr -d '\n')
# Prompt creation for subjective dimentions creation
task_content=$(tr -d '\r' < ./dimension_task/$dimension/$version/$criteria.txt | tr -d '\n')


# Use null-terminated output from find to handle file names with spaces
find transcripts -type f -name "*.txt" -print0 | while IFS= read -r -d '' transcript_file; do

    # Extract information from the transcript file path
    relative_path_without_prefix=${transcript_file#transcripts/}  # Remove the prefix
    transcript_id=$(basename "$relative_path_without_prefix" .txt)


    # Combine texts from task.txt and the current transcriptXXX-&&-category.txt file
    # constructed_prompt="TRANSCRIPTION = [ $(cat "$transcript_file" | tr -d '\r' | tr -d '\n') ]  $task_content "
    constructed_prompt="TRANSCRIPT = [ $(cat "$transcript_file" | tr -d '\r' | tr -d '\n') ] $task_content "

    # Create subdirectories in prompts folder based on the structure of transcripts_whisper/
    # output_folder="prompts/$dimension/$criteria/$version" # Updated output folder path
    output_folder="prompts/$dimension/$version" # Updated output folder path
    mkdir -p "$output_folder"

    # Save the constructed prompt to prompts/promptXXX-category.txt
    output_file="$output_folder/${transcript_id}.txt"  # Remove spaces from transcript_base
    echo "$constructed_prompt" > "$output_file"
    echo "Constructed prompt saved to: $output_file"
done
