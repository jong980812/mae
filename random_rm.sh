# Specify the directory to remove PNG files from
folder_path="/local_datasets/quickdraw/subset_train/angel_temp"  # Replace with the actual path to your folder
# Validate the input
if ! [ -d "$folder_path" ]; then
  echo "Error: Directory \"$folder_path\" does not exist."
  exit 1
fi

# Change to the specified directory
cd "$folder_path"

# Find all PNG files in the directory and store them in an array
png_files=(*.png)

# Check if the array is empty (i.e., no PNG files found)
if [ "${#png_files[@]}" -eq 0 ]; then
  echo "No PNG files found in \"$folder_path\"."
  exit 0
fi

# Input the number of files to remove
read -p "Enter the number of PNG files to remove (more than 10): " num_files_to_remove

# Validate the input
if ! [[ "$num_files_to_remove" =~ ^[0-9]+$ ]]; then
  echo "Error: Please enter a valid number."
  exit 1
fi

# Check if the number of PNG files is less than the number of files to remove
if [ "${#png_files[@]}" -le "$num_files_to_remove" ]; then
  echo "Number of PNG files in \"$folder_path\" is less than or equal to $num_files_to_remove. No files will be removed."
  exit 0
fi

# Shuffle the array of PNG files randomly
shuffled_files=($(shuf -e "${png_files[@]}"))

# Remove the specified number of files from the shuffled array
files_to_remove=("${shuffled_files[@]:0:num_files_to_remove}")

# Remove the selected files one by one
for file in "${files_to_remove[@]}"; do
  rm "$file"
  echo "Removed file: $file"
done