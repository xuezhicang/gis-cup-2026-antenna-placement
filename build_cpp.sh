#!/usr/bin/env bash
# Compile the three C++ extensions into ./build so the Python steps can import them.
#
#   bash build_cpp.sh                 # uses "python3"
#   PYTHON=/path/to/python bash build_cpp.sh
#
# Needs a C++17 compiler and the pybind11 Python package (pip install pybind11).
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
CXX="${CXX:-c++}"
BUILD_DIR="build"

EXT_SUFFIX="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
PYBIND_INCLUDES="$("$PYTHON" -m pybind11 --includes)"

LINK_FLAGS=(-shared)
if [[ "$(uname -s)" == "Darwin" ]]; then
    # macOS resolves Python symbols when the module is loaded.
    LINK_FLAGS+=(-undefined dynamic_lookup)
fi

mkdir -p "$BUILD_DIR"
for module in coverage_scorer_cpp visibility_tree_cpp building_kdtree_cpp; do
    case "$module" in
        coverage_scorer_cpp) source_file="cpp/coverage_scorer_cpp.cpp" ;;
        visibility_tree_cpp) source_file="cpp/visibility_tree_cpp.cpp" ;;
        building_kdtree_cpp) source_file="cpp/building_kdtree.cpp" ;;
    esac
    echo "Building $module"
    # shellcheck disable=SC2086
    "$CXX" -O3 -Wall -std=c++17 -fPIC $PYBIND_INCLUDES -Icpp \
        "$source_file" "${LINK_FLAGS[@]}" -o "$BUILD_DIR/${module}${EXT_SUFFIX}"
done

"$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, "build")
import building_kdtree_cpp, coverage_scorer_cpp, visibility_tree_cpp  # noqa: F401
print("All C++ extensions built and importable.")
EOF
