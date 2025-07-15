#!/bin/bash

# Create a directory for terminal output if it doesn't exist
mkdir -p terminaloutput

model_name="llama3" #130 gb videomemory
dimension="creativity"
criteria_group="storytelling"
criteria="metaphor"
version_prompt="4"
version="1"

# Check if there are prompt files in prompts/ directory and its subfolders
# prompt_files=($(find prompts/$criteria_group/$criteria/$version_prompt/ -type f -name "*.txt"))
# Analysis of subjective dimensions
prompt_files=($(find prompts/$dimension/$version_prompt/ -type f -name "*.txt"))

if [ ${#prompt_files[@]} -gt 0 ]; then
    # Iterate over found prompt files

    for prompt_file in "${prompt_files[@]}"; do
        # Extract id of the prompt
        # prompt_name=${prompt_file#prompts/$criteria_group/$criteria/$version_prompt/}  # Remove the prefix
        prompt_name=${prompt_file#prompts/$dimension/$version_prompt/}  # Remove the prefix
        prompt_id=$(basename "$prompt_name" .txt)
        
        # Read content from the current promptXXX&&-category.txt file, removing any carriage return characters
        prompt_content=$(tr -d '\r' < "$prompt_file" | tr -d '\n')
        # echo $prompt_content

        # Create subdirectories in terminaloutput folder based on the structure of prompts/
        # output_folder="terminaloutput/$criteria_group/$criteria/$version_prompt/$model_name-$version"
        output_folder="terminaloutput/$dimension/$version_prompt/$model_name-$version"

        mkdir -p "$output_folder"

        # Create a JSON file with the payload
        echo "{\"model\": \"$model_name\", \"prompt\": \"Combining texts: $prompt_content\", \"stream\": false}" > payload.json

        # Use jq to read the JSON file and pass it to curl, saving the output as a JSON file with the corresponding name
        curl -H "Content-Type: application/json" -X POST -d @payload.json "http://localhost:11434/api/generate" > "$output_folder/$prompt_id.json"

        # Remove the temporary JSON file
        rm payload.json
        
    done
else
    echo "No prompt files found in prompts/ directory or its subdirectories."
fi
