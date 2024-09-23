from subprocess import PIPE, Popen

while True:
    with Popen(
        "python main.py", stdout=PIPE, shell=True, universal_newlines=True
    ) as main_p:
        for stdout_line in iter(main_p.stdout.readline, ""):
            print(stdout_line)
        # pylint: disable=invalid-name
        return_code = main_p.wait()
        if return_code:
            print("ERROR")
            for line in iter(main_p.stderr.readline, ""):
                print(line)
        print("======")
