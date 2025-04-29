with open('main.py', 'r') as f:
    lines = f.readlines()

# Find the start of the main function
main_start = -1
for i, line in enumerate(lines):
    if line.strip() == 'def main():':
        main_start = i
        break

if main_start == -1:
    print("Could not find main function")
    exit(1)

# Find the section with the legend display code
legend_start = -1
for i in range(main_start, len(lines)):
    if "# Display legend" in lines[i]:
        legend_start = i
        break

if legend_start == -1:
    print("Could not find legend display code")
    exit(1)

# Find the end of the if block
if_end = -1
for i in range(legend_start, len(lines)):
    if lines[i].strip() == "else:":
        if_end = i
        break

if if_end == -1:
    print("Could not find end of if block")
    exit(1)

# Find the misplaced legend code
misplaced_start = -1
for i in range(if_end, len(lines)):
    if "with legend_cols[0]:" in lines[i]:
        misplaced_start = i
        break

if misplaced_start == -1:
    print("Could not find misplaced legend code")
    exit(1)

# Find the end of the misplaced legend code
misplaced_end = -1
for i in range(misplaced_start, len(lines)):
    if "# Create and display heatmap" in lines[i]:
        misplaced_end = i
        break

if misplaced_end == -1:
    print("Could not find end of misplaced legend code")
    exit(1)

# Find the end of the second heatmap code
second_heatmap_end = -1
for i in range(misplaced_end, len(lines)):
    if "# Display raw data" in lines[i]:
        second_heatmap_end = i
        break

if second_heatmap_end == -1:
    print("Could not find end of second heatmap code")
    exit(1)

# Extract the misplaced legend code
misplaced_code = lines[misplaced_start:misplaced_end]

# Fix the indentation of the misplaced code
fixed_code = []
for line in misplaced_code:
    # Remove leading whitespace and add proper indentation (12 spaces)
    fixed_code.append("            " + line.lstrip())

# Insert the fixed code after the legend_cols declaration
insert_pos = legend_start + 2  # After "legend_cols = st.columns(7)"

# Remove the misplaced code
del lines[misplaced_start:misplaced_end]

# Insert the fixed code
for i, line in enumerate(fixed_code):
    lines.insert(insert_pos + i, line)

# Remove the duplicate heatmap code
heatmap_start = -1
for i in range(insert_pos + len(fixed_code), len(lines)):
    if "# Create and display heatmap" in lines[i]:
        heatmap_start = i
        break

if heatmap_start != -1:
    # Find the end of the duplicate heatmap code
    heatmap_end = -1
    for i in range(heatmap_start, len(lines)):
        if "# Display raw data" in lines[i]:
            heatmap_end = i
            break
    
    if heatmap_end != -1:
        # Remove the duplicate heatmap code
        del lines[heatmap_start:heatmap_end]

# Write the fixed content back to the file
with open('main.py', 'w') as f:
    f.writelines(lines)

print("Fixed main.py")
