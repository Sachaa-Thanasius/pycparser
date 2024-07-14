import os.path


def main() -> None:
    for cur_path, _, files in os.walk("."):
        if cur_path == ".":
            for f in files:
                if f.endswith(".h"):
                    print(f)
                    with open(f, "w") as fo:
                        fo.write('#include "_fake_defines.h"\n')
                        fo.write('#include "_fake_typedefs.h"\n')


if __name__ == "__main__":
    raise SystemExit(main())
