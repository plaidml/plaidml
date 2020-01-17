import glob


def fix_doxyxml(pattern):
    for xml in glob.glob(pattern):
        with open(xml, "r") as xmlfile:
            orig_lines = xmlfile.readlines()
        with open(xml, "w") as xmlfile:
            for line in orig_lines:
                if "</includes>" not in line:
                    xmlfile.write(line)
