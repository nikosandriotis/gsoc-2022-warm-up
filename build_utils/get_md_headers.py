import sys
import os

import build_utils

url = "https://github.com/intel/metrics-discovery.git"
commit = "7c416548fe13274effb795fcee55d099aee41b3f"

def main():
  if len(sys.argv) < 3:
    print("Usage: python get_md_headers.py <include_path> <build_path>")
    return

  dst_path = sys.argv[1]
  if (not os.path.exists(dst_path)):
    os.mkdir(dst_path)

  clone_path = sys.argv[2]
  clone_path = os.path.join(clone_path, "metrics-discovery")
  build_utils.clone(url, commit, clone_path)

  src_path = os.path.join(clone_path, "inc")
  src_path = os.path.join(src_path, "common")
  src_path = os.path.join(src_path, "instrumentation")
  src_path = os.path.join(src_path, "api")

  build_utils.copy(src_path, dst_path, ["metrics_discovery_api.h"])

if __name__ == "__main__":
  main()