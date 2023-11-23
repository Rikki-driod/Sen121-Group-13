bench_pr=float(input("What is your bench press pr (kg): "))
squat_pr=float(input("What is your squat pr (kg): "))
deadlift_pr=float(input("What is your deadlift pr (kg): "))
total=bench_pr+squat_pr+deadlift_pr

if (bench_pr>=100 and bench_pr<130) and (squat_pr>=140 and squat_pr<230) and (deadlift_pr>=180 and deadlift_pr<280):
    print("man")
elif bench_pr>=130 and squat_pr>=230 and deadlift_pr>=280:
    print("Sam Sulek")
else:
    print("weak")

print(total)