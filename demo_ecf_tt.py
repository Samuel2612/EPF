
# Demo runner for ecf_tt.py
from ecf_tt import demo_small

if __name__ == "__main__":
    stats = demo_small()
    print("ECF TT demo (small)")
    print("  grid shape:", stats["shape"])
    print("  TT ranks  :", stats["ranks"])
    print(f"  RMSE      : {stats['rmse']:.3e}")
    print(f"  Max error : {stats['max_err']:.3e}")
