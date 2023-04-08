"""

Main function to classifaction with differnt images

File: test_with_set_v_main.py
Author: Zoe LaLena
Date: 4/7/2023
Course: Senior Project

"""
import test_with_set_v

def main():
    set_v_file = "318r_V_refl_2500_4100.csv"

    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/900_4900", "900_4900")
    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/900_5000", "900_5000")
    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/2500_4100", "2500_4100")
    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/3000_2500", "3000_2500")
    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/3100_3000", "3100_3000")
    test_with_set_v.testWithSetV(set_v_file, "imageFiles/318r/3400_3100", "3400_3100")



if __name__ == '__main__':
    main()