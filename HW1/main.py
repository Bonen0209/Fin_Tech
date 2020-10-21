import numpy as np
import argparse
from linear_regression import Linear_Regression


args = {
      'data_dir': '../Data/hw1/',
      'output_dir': './Results/'
}
args = argparse.Namespace(**args)

def main():
    print('Student')
    LR_student = Linear_Regression(data_dir=args.data_dir, category='student', output_dir=args.output_dir)
    
    X_test = np.load(f'{args.data_dir}student_test_no_G3_x.npy')
    X_test_with_1 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    Y_hat_test = LR_student.predict(X_test_with_1)


    with open(f'{args.output_dir}r08942073_1.txt', 'w') as f:
        for idx in range(len(Y_hat_test)):
            f.write(str(1001 + idx))
            f.write('\t')
            f.write('{:.1f}'.format(Y_hat_test[idx]))
            f.write('\n')

    print('Census')
    LR_census = Linear_Regression(data_dir=args.data_dir, category='census', output_dir=args.output_dir)

if __name__ == '__main__':
    main()
