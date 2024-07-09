#!/bin/bash

# Define the Apache 2.0 License header
LICENSE_HEADER="Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an \"AS IS\" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"

# Function to add license to a file
add_license_to_file() {
    local file=$1
    if ! grep -q "Copyright 2024 Google LLC" "$file"; then
        echo -e "$LICENSE_HEADER
$(cat "$file")" > "$file"
    fi
}

# Export the function for parallel processing
export -f add_license_to_file
export LICENSE_HEADER

# Find all source files and add the license header
find . -type f -name '*.py' -exec bash -c 'add_license_to_file "$0"' {} \;
