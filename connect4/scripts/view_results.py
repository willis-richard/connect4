import sys
import pstats
from pstats import SortKey
p = pstats.Stats(sys.argv[1])
#p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
#p.sort_stats(SortKey.TIME).print_stats(10)
p.print_callers(10)
