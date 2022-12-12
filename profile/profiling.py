import cProfile
import subprocess
import code

code.prepare(n_samples=1000)
code.start()
cProfile.run('code.start()', filename='output.prof')
subprocess.run('snakeviz output.prof')
