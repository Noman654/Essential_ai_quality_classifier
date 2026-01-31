#!/bin/bash
set -e

# --- Configuration ---
BASE_INPUT_DIR="nemov2/"
OUTPUT_BASE_DIR="./quality_classified"
SCRIPT_PATH="./main.py"
# SCRIPT_PATH="./test2_backup_optimized.py"


# List of IPs
IPS=(
    "ip"
    "ip1"
    "localhost"
)

# Clean up previous task lists
rm -f task_list_*.txt
mkdir -p logs

# Get list of all parquet files
echo "Finding files..."
FILES=($(find -L "$BASE_INPUT_DIR" -type f -name "*.parquet" | sort))
NUM_FILES=${#FILES[@]}
NUM_IPS=${#IPS[@]}

echo "Found $NUM_FILES files to process using $NUM_IPS IPs."

# Distribute files to task lists (Round Robin)
echo "Distributing tasks per IP..."
for ((i=0; i<NUM_FILES; i++)); do
    FILE="${FILES[$i]}"
    IP_INDEX=$((i % NUM_IPS))
    IP="${IPS[$IP_INDEX]}"
    echo "$FILE" >> "task_list_${IP}.txt"
done

# Function to process files for a specific IP
process_ip_tasks() {
    local IP=$1
    local TASK_FILE="task_list_${IP}.txt"
    
    if [ ! -f "$TASK_FILE" ]; then
        echo "[$IP] No tasks assigned."
        return
    fi
    
    local COUNT=0
    local TOTAL=$(wc -l < "$TASK_FILE")
    
    echo "[$IP] Starting worker... ($TOTAL items)"
    
    while IFS= read -r INPUT_FILE; do
        COUNT=$((COUNT + 1))
        
        # Prepare Output Path
        SUBDIR=$(basename $(dirname "$INPUT_FILE"))
        FILENAME=$(basename "$INPUT_FILE" .parquet)
        OUTPUT_FILE="$OUTPUT_BASE_DIR/$SUBDIR/${FILENAME}_classified.parquet"
        mkdir -p "$OUTPUT_BASE_DIR/$SUBDIR"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "[$IP] [$COUNT/$TOTAL] Skipping: $(basename "$INPUT_FILE") (Done)"
            continue
        fi
        
        echo "[$IP] [$COUNT/$TOTAL] Processing: $(basename "$INPUT_FILE") -> Pending..."
        
        # Run Python Script
        # We capture python output to a specific log file
        LOG_FILE="logs/${FILENAME}_${IP}.log"
        
        if python3 "$SCRIPT_PATH" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --api_ip "$IP" > "$LOG_FILE" 2>&1; then
            echo "[$IP] [$COUNT/$TOTAL] ✓ Finished: $(basename "$INPUT_FILE")"
        else
            echo "[$IP] [$COUNT/$TOTAL] ✗ Failed: $(basename "$INPUT_FILE") (Check $LOG_FILE)"
        fi
        
    done < "$TASK_FILE"
    
    echo "[$IP] All tasks completed."
}

# Export function and variables for subshells (not strictly needed if running in same shell loop, but good practice)
export BASE_INPUT_DIR OUTPUT_BASE_DIR SCRIPT_PATH

# Launch background workers for each IP
echo "Launching $NUM_IPS worker processes..."
for IP in "${IPS[@]}"; do
    process_ip_tasks "$IP" &
done

# Wait for all workers to finish
wait
echo "Distributed processing finished!"
rm -f task_list_*.txt
