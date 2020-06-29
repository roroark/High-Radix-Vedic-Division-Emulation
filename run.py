from math import *

def add_signed_binary_numbers(b1, b2):
    try:
        b1.index(".")
    except ValueError:
        b1 = b1 + "."
    try:
        b2.index(".")
    except ValueError:
        b2 = b2 + "."
    b1_fraction_len = "".join(reversed(b1)).index(".")
    b2_fraction_len = "".join(reversed(b2)).index(".")
    if b1_fraction_len > b2_fraction_len:
        b2 = b2 + "0" * (b1_fraction_len - b2_fraction_len)
    elif b2_fraction_len > b1_fraction_len:
        b1 = b1 + "0" * (b2_fraction_len - b1_fraction_len)
    rev_b1 = "".join(reversed(b1))
    rev_b2 = "".join(reversed(b2))
    rev_sum = ""
    carry_in = 0
    carry_out = 0
    i = 0
    while i < max(len(b1), len(b2)) or carry_in != carry_out:
        carry_in = carry_out
        rev_b1_i = rev_b1[min(i, len(b1) - 1)]
        rev_b2_i = rev_b2[min(i, len(b2) - 1)]
        if rev_b2_i == "." and rev_b1_i == ".":
            rev_sum += "."
        else:
            rev_b1_i = int(rev_b1_i)
            rev_b2_i = int(rev_b2_i)
            rev_sum += str((rev_b1_i + rev_b2_i + carry_in) % 2)
            if rev_b2_i + rev_b1_i + carry_in > 1:
                carry_out = 1
            else:
                carry_out = 0

        i += 1
    sum = "".join(reversed(rev_sum))
    while sum.startswith("00"):
        sum = sum[1:]
    while sum.startswith("11"):
        sum = sum[1:]
    if sum.endswith("."):
        sum = sum[:-1]

    return sum

def complement_binary_number(b1):
    rev_b1 = "".join(reversed(b1))
    rev_complement = ""
    one_encountered = False
    for i in range(len(rev_b1)):
        if one_encountered:
            if rev_b1[i] == "1":
                rev_complement += "0"
            elif rev_b1[i] == "0":
                rev_complement += "1"
            else:
                rev_complement += rev_b1[i]
        else:
            rev_complement += rev_b1[i]
            if rev_b1[i] == "1":
                one_encountered = True
                if i + 1 == len(rev_b1):
                    rev_complement += "0"


    complement = "".join(reversed(rev_complement))
    return complement

# Adds three ORDINARY numbers and returns the sum in stored-carry form
def carry_save_add_binary_numbers(b1, b2, b3):
    try:
        b1.index(".")
    except ValueError:
        b1 = b1 + "."
    try:
        b2.index(".")
    except ValueError:
        b2 = b2 + "."
    try:
        b3.index(".")
    except ValueError:
        b3 = b3 + "."
    b1_fraction_len = "".join(reversed(b1)).index(".")
    b2_fraction_len = "".join(reversed(b2)).index(".")
    b3_fraction_len = "".join(reversed(b3)).index(".")
    if b1_fraction_len < max(b1_fraction_len, b2_fraction_len, b3_fraction_len):
        b1 = b1 + "0" * (max(b1_fraction_len, b2_fraction_len, b3_fraction_len) - b1_fraction_len)
    if b2_fraction_len < max(b1_fraction_len, b2_fraction_len, b3_fraction_len):
        b2 = b2 + "0" * (max(b1_fraction_len, b2_fraction_len, b3_fraction_len) - b2_fraction_len)
    if b3_fraction_len < max(b1_fraction_len, b2_fraction_len, b3_fraction_len):
        b3 = b3 + "0" * (max(b1_fraction_len, b2_fraction_len, b3_fraction_len) - b3_fraction_len)

    rev_b1 = "".join(reversed(b1))
    rev_b2 = "".join(reversed(b2))
    rev_b3 = "".join(reversed(b3))
    rev_o1 = ""
    rev_o2 = ""
    # print(b1)
    # print(b2)
    # print(b3)
    for i in range(max(len(b1), len(b2), len(b3))):
        rev_b1_i = rev_b1[min(i, len(b1) - 1)]
        rev_b2_i = rev_b2[min(i, len(b2) - 1)]
        rev_b3_i = rev_b3[min(i, len(b3) - 1)]
        if rev_b1_i == ".":
            rev_o1 = rev_o1 + "."
            rev_o2 = rev_o2 + "."
            continue
        else:
            rev_b1_i = int(rev_b1_i)
            rev_b2_i = int(rev_b2_i)
            rev_b3_i = int(rev_b3_i)
            # print("#"+str(i))
            # print(rev_b1_i)
            # print(rev_b2_i)
            # print(rev_b3_i)
            # print (str((rev_b1_i + rev_b2_i + rev_b3_i) % 2))
            # print (str(int((rev_b1_i + rev_b2_i + rev_b3_i) > 1)))
            rev_o1 = rev_o1 + str((rev_b1_i + rev_b2_i + rev_b3_i) % 2)
            rev_o2 = rev_o2 + str(int((rev_b1_i + rev_b2_i + rev_b3_i) > 1))

    o1 = "".join(reversed(rev_o1))
    o2 = left_shift_binary_number("".join(reversed(rev_o2)), 1)

    # print ("##")
    # print (o1)
    # print (o2)

    while o1.startswith("00"):
        o1 = o1[1:]
    while o1.startswith("11"):
        o1 = o1[1:]
    if o1.endswith("."):
        o1 = o1[:-1]
    while o2.startswith("00"):
        o2 = o2[1:]
    while o2.startswith("11"):
        o2 = o2[1:]
    if o2.endswith("."):
        o2 = o2[:-1]

    return [o1, o2]

#   Converts an integer to two's complement binary string where the first character / bit is the sign.
def integer_to_signed_binary(integer):
    if integer >= 0:
        return "0" + "{0:b}".format(integer)
    else:
        return add_signed_binary_numbers(
            ("0" + "{0:b}".format(-integer)).replace("1", "2").replace("0", "1").replace("2", "0"), "01")

def signed_binary_to_integer(binary):
    try:
        binary.index(".")
    except ValueError:
        binary = binary + "."

    before_decimal = binary[:binary.index(".")]
    after_decimal  = binary[binary.index(".")+1:]

    #Before Decimal
    rev_input_binary = "".join(reversed(before_decimal))
    two_pow = 1
    input_decimal = 0
    for i in range(len(rev_input_binary)):
        if i + 1 == len(rev_input_binary) and rev_input_binary[i] == "1":
            input_decimal += -two_pow
        elif rev_input_binary[i] == "1":
            input_decimal += two_pow
        two_pow = two_pow * 2

    two_pow = 0.5
    #After Decimal
    for i in range(len(after_decimal)):
        if after_decimal[i] == "1":
            input_decimal += two_pow
        two_pow = two_pow * 0.5
    return input_decimal

#   normalize_binary_divisor
#   Asumes below, which are ensured at the output of 'integer_to_signed_binary':
#       'd' starts with '10' or '01'
#       No decimal point.
def normalize_binary_divisor(base, d):
    #Special case of -1
    if d == "1":
        d = "11"

    normalized_divisor = d[0:2] + "." + d[2:]
    base_bit_count = int(log2(base))
    first_digit_with_sign = "".join(
        reversed("".join(reversed(d))[((len(d) - 1) // base_bit_count - 1) * base_bit_count:]))
    dividend_right_shift_amount = len(first_digit_with_sign) - 2
    # Special Case for negative powers of 2:
    if (d[2:] == "" or int(d[2:]) == 0) and d[0:2] == "10":
        normalized_divisor = left_shift_binary_number(normalized_divisor,-1)
        dividend_right_shift_amount += 1

    return [normalized_divisor, dividend_right_shift_amount]


#   Left shift a binary number by certain number of bits.
#   Negative values of amount = right shift
def left_shift_binary_number(b1, amount):
    if amount == 0:
        return b1
    try:
        b1.index(".")
    except ValueError:
        b1 = b1 + "."

    if amount > 0:
        b1 = b1 + "0" * amount

    decimal_index = b1.index(".")
    before_decimal = b1[:decimal_index]
    after_decimal = b1[decimal_index + 1:]
    whole_number = before_decimal + after_decimal
    if amount < 0:
        # 011.10 >> 4 = 0.001110
        # decimal_index = 3, shift_amount = -4: sign_extention = 3+(-4)+1
        # 011.10 >> 1  = 01.110
        # decimal_index = 3, shift_amount = 3:
        if decimal_index + amount < 1:
            sign = before_decimal[0]
            shifted_number = sign + "." + sign * (-amount - decimal_index) + whole_number
        else:
            shifted_number = whole_number[:decimal_index + amount] + "." + whole_number[decimal_index + amount:]
    else:
        # We had padded zeros at the end earlier.
        shifted_number = whole_number[:decimal_index + amount] + "." + whole_number[decimal_index + amount:]
        while shifted_number.endswith("0"):
            shifted_number = shifted_number[:-1]
        if shifted_number.endswith("."):
            shifted_number = shifted_number[:-1]
    return shifted_number


#   Normalize and right shift the dividend.
def normalize_binary_dividend(base, x):
    while x.startswith("00") or x.startswith("11"):
        x = x[1:]
    try:
        x.index(".")
    except ValueError:
        x = x + "."
    # First normalize based on bit grouping such that |x| is in [0.1,1) in the corresponding base.
    before_decimal = x[:x.index(".")]
    after_decimal = x[x.index(".") + 1:]
    base_bit_count = int(log2(base))
    sign = x[0]
    first_digit_with_sign = "".join(
        reversed("".join(reversed(before_decimal))[((len(before_decimal) - 1) // base_bit_count) * base_bit_count:]))
    remaining_digits = "".join(
        reversed("".join(reversed(before_decimal))[:((len(before_decimal) - 1) // base_bit_count) * base_bit_count]))
    # Complete first digit (if needed).
    while (len(first_digit_with_sign) - 1) % base_bit_count != 0:
        first_digit_with_sign = sign + first_digit_with_sign
    # Currently the dividend is in the form 0D1D2.. or 1D1D2... where D1 and D2 are digits in the corresponding base.
    # Additional right shifting if pre-scaling the divisor.
    normalized_dividend = first_digit_with_sign[0] + "." + first_digit_with_sign[1:] + remaining_digits + after_decimal
    return normalized_dividend


#   Note:
#   b1  b2 is ORDINARY NON-NEGATIVE binary number.
#   returns product in stored carry form.
#   If the base is fixed, every call actually has a fixed length b2, fixing implementation size.
#   The tree can be further parallelized in hardware.
def carry_save_multiply_binary_numbers(b1, b2):
    # Asserting positive b2.
    assert (b2[0] == "0")
    try:
        b2.index(".")
    except ValueError:
        b2 = b2 + "."
    product = ["0", "0"]
    decimal_index = b2.index(".")
    for i in range(len(b2)):
        if i == decimal_index or b2[i] == "0":
            continue
        elif i < decimal_index:
            left_shift_amount = (decimal_index - 1) - i
            product = carry_save_add_binary_numbers(product[0], product[1],
                                                    left_shift_binary_number(b1, left_shift_amount))
        else:
            right_shift_amount = i - decimal_index
            product = carry_save_add_binary_numbers(product[0], product[1],
                                                    left_shift_binary_number(b1, -right_shift_amount))
    return product


# Pre-scaling to [1-2/base].
# Need 2 additional bits for radix-4 and 3 additional bits for radix-8.
def prescale_factor(base, d_n):
    d_n = d_n + "0" * base
    if d_n.index(".") == 1:
        d_n = d_n[0] + d_n
    if base == 4:
        if d_n.startswith("01.1") or d_n.startswith("10.0"):
            return "01.0000"
        elif d_n.startswith("01.01") or d_n.startswith("10.10"):
            return "01.0100"
        elif d_n.startswith("01.00") or d_n.startswith("10.11"):
            return "01.1000"
        else:
            raise ValueError("ERROR: Could not pre-scale divisor " + str(d_n) + " in base " + str(base) + ".")
    elif base == 8:
        if d_n.startswith("01.11") or d_n.startswith("10.00"):
            return "01.0000"
        elif d_n.startswith("01.101") or d_n.startswith("10.010"):
            return "01.0010"
        elif d_n.startswith("01.100") or d_n.startswith("10.011"):
            return "01.0011"
        elif d_n.startswith("01.011") or d_n.startswith("10.100"):
            return "01.0101"
        elif d_n.startswith("01.010") or d_n.startswith("10.101"):
            return "01.0111"
        elif d_n.startswith("01.001") or d_n.startswith("10.110"):
            return "01.1001"
        elif d_n.startswith("01.000") or d_n.startswith("10.111") or d_n.startswith("11.000"):
            return "01.1100"
        else:
            raise ValueError("ERROR: Could not pre-scale divisor " + str(d_n) + " in base " + str(base) + ".")
    else:
        raise ValueError("ERROR: Base " + str(base) + " pre-scaling factors not coded.")


#   Performs full-carry-propagate on a carry-save input.
def carry_propagate(b_cs):
    return add_signed_binary_numbers(b_cs[0], b_cs[1])


#   Returns extent of valid digits after the decimal point. This can include zeros.
def num_digits_dividend(base, r_in):
    base_bit_count = int(log2(base))
    after_decimal = r_in[r_in.index(".") + 1:]
    return ceil(((len(after_decimal) - 1) / base_bit_count))


# Add additional
def num_digits_divisor(base, d_p):
    try:
        d_p.index(".")
    except ValueError:
        d_p = d_p + "."
    base_bit_count = int(log2(base))
    after_decimal = d_p[d_p.index(".") + 1:]
    return 1 + ceil(((len(after_decimal) - 1) / base_bit_count))


#   normalize_input(base,X,D)
#   Returns the parameters needed to perform the division iterations depending on base.
def normalize_input(base, X, D, debug):
    base_bit_count = int(log2(base))
    # Normalized divisor is base-independent but can require the dividend to be right-shifted (pre-scaled) as the first
    # digit may not be 1.
    [d_n, dividend_right_shift_amount] = normalize_binary_divisor(base, integer_to_signed_binary(D))
    p = prescale_factor(base, d_n)
    d_p = carry_propagate(carry_save_multiply_binary_numbers(d_n, p))
    assert (signed_binary_to_integer(p)*signed_binary_to_integer(d_n)==signed_binary_to_integer(d_p))
    # Right shift by the amount needed + base_bit_count as pre-scaling the dividend and divisor later can result in
    # |r_in_cs| >= 1 which breaks normalization.
    x_n = normalize_binary_dividend(base,
                                  left_shift_binary_number(integer_to_signed_binary(X), -dividend_right_shift_amount))
    # Pre-scaling

    # Carry propagate if pre-scaling to compute n and m.
    # If not pre-scaling, we needn't perform multiplication, hence inputs remain in carry-propagated form.
    # r_in
    x_p = carry_propagate(carry_save_multiply_binary_numbers(x_n, p))
    assert (signed_binary_to_integer(p) * signed_binary_to_integer(x_n) == signed_binary_to_integer(x_p))
    r_in = normalize_binary_dividend(base, x_p)
    pc_r_in = r_in
    if D < 0:
        r_in = normalize_binary_dividend(base,complement_binary_number(r_in))
    #r_in_list = list(r_in)
    #r_in_list[0] = str(int(X < 0) ^ int(D < 0))
    #r_in = "".join(r_in_list)
    #p_in = str(int(X < 0))
    p_in = str(int(X < 0) ^ int(D < 0))
    #p_in = "0"
    # Computing the length of the pre-scaled operands requires a full-carry-propagate.
    # Note: These include zeros which may have been right shifted beyond the decimal point which we need to track.
    n = num_digits_dividend(base, r_in)
    m = num_digits_divisor(base, d_p)

    if debug:
        print("Dividend: " + str(X) + " Divisor: " + str(D))
        print("Normalized dividend before pre-scaling: " + x_n)
        print("Normalized divisor before pre-scaling: " + d_n)
        print("Pre-scaling factor: " + p)
        print("r_in': " + pc_r_in)
        print("r_in: " + r_in + " n: " + str(n))
        print("Normalized divisor: " + d_p + " m: " + str(m))
        print("pin: " + p_in)

    return [r_in, p_in, d_p, n, m]


def radix8_prescaled_pla_access(input):
    rev_input_binary = "".join(reversed(input))
    two_pow = 1
    input_decimal = 0
    for i in range(len(rev_input_binary)):
        if i + 1 == len(rev_input_binary) and rev_input_binary[i] == "1":
            input_decimal += -two_pow
        elif rev_input_binary[i] == "1":
            input_decimal += two_pow
        two_pow = two_pow * 2

    if  input_decimal >= 12:
        # -7
        return "1001"
    elif input_decimal >= 10:
        # -6
        return "1010"
    elif input_decimal >= 8:
        # - 5
        return "1011"
    elif input_decimal >= 6:
        # -4
        return "1100"
    elif input_decimal >= 4:
        # -3
        return "1101"
    elif input_decimal >= 2:
        # -2
        return "1110"
    elif input_decimal >= 1:
        # -1
        return  "1111"
    elif input_decimal >= -1:
        # 0
        return "0000"
    elif input_decimal >= -2:
        # +1
        return "0001"
    elif input_decimal >= -4:
        # +2
        return "0010"
    elif input_decimal >= -6:
        # +3
        return "0011"
    elif input_decimal >= -8:
        # +4
        return "0100"
    elif input_decimal >= -10:
        # +5
        return "0101"
    elif input_decimal >= -12:
        # +6
        return "0110"
    else:
        # +7
        return "0111"


def divide_stage(base, r_in, p_in, d_p, n, m, i, debug):
    if i > n - m + 1:
        return [r_in, p_in]
    else:
        base_bit_count = int(log2(base))
        # Compute o
        # Look at [p_in,r_in[0:]]<<log2(base)
        pla_input = p_in + left_shift_binary_number(r_in[r_in.index("."):r_in.index(".") + base_bit_count + 1], base_bit_count)
        o = radix8_prescaled_pla_access(pla_input)
        #Complement d_p if d_p<0. This is needed because we have not coded the second and third quadrant of PLA.
        if d_p[0] == "1":
            d_p = complement_binary_number(d_p)
        # Workaround for not having a negative multiplier.
        #################################################
        if o[0] == "1":
            o = complement_binary_number(o)
            complemented_o = True
        else:
            complemented_o = False
        #################################################
        # Compte r_out
        fractional_d_p = d_p
        fractional_d_p_list = list(fractional_d_p)
        fractional_d_p_list[1] = fractional_d_p_list[0]
        fractional_d_p = "".join(fractional_d_p_list)
        fdm = carry_propagate(carry_save_multiply_binary_numbers(fractional_d_p, o))
        # Workaround for not having a negative multiplier.
        #################################################
        if complemented_o:
            o = complement_binary_number(o)
            fdm = complement_binary_number(fdm)
        #################################################
        # Subtract multiple of divisor from r_in.
        r_out = add_signed_binary_numbers(left_shift_binary_number(r_in,base_bit_count), fdm)
        # Compute p_out
        try:
            r_out.index(".")
        except ValueError:
            r_out = r_out + "."

        p_out = add_signed_binary_numbers(p_in + r_out[max(0,r_out.index(".") - base_bit_count):r_out.index(".")], o)[-2:]
        if debug:
            print("################")
            print("Iteration #" + str(i))
            print("Shifted r_in: " + left_shift_binary_number(r_in,base_bit_count) + " (" + str(signed_binary_to_integer(left_shift_binary_number(r_in,base_bit_count))) + ") "+ " p_in: " + p_in)
            print("PLA Input: " + pla_input + " (" + str(signed_binary_to_integer(pla_input)) + ") " + " Operation: " + str(o) + " (" + str(signed_binary_to_integer(o)) + ")")
            print("Fractional-divisor-multiple: " + fdm + " (" + str(signed_binary_to_integer(fdm)) + ")")
            print("r_out: " + r_out + " p_out: " + p_out)
            print("r_out(real: " + add_signed_binary_numbers(r_out,o))
            print("#p_out:")
            print("#" + (p_in + r_out[max(0, r_out.index(".") - base_bit_count):r_out.index(".")])[-2:])
            print("#" + o)

    return [r_out, p_out]


def adjust_result(r_in, p_in, d_p, debug):
    quotient_corrected = False
    if len(p_in) == 1:
        p_in += p_in
    if p_in == "01":
        r_in = add_signed_binary_numbers(r_in,"11")
        quotient_corrected = True
    elif p_in == "10":
        r_in = add_signed_binary_numbers(r_in,"01")
        quotient_corrected = True
    try:
        r_in.index(".")
    except ValueError:
        r_in = r_in + "."
    quotient_binary = r_in[0:r_in.index(".")]
    # Convert to decimal
    rev_quotient_binary = "".join(reversed(quotient_binary))
    two_pow = 1
    quotient_decimal = 0
    for i in range(len(rev_quotient_binary)):
        if i + 1 == len(rev_quotient_binary) and rev_quotient_binary[i] == "1":
            quotient_decimal += -two_pow
        elif rev_quotient_binary[i] == "1":
            quotient_decimal += two_pow
        two_pow = two_pow * 2

    if debug:
        print("##############")
        print("End of Iterations")
        print("Quotient: " + str(quotient_decimal))
        print("Quotient was corrected: " + str(quotient_corrected))

    return quotient_decimal


# base, X, D in decimal.
def vedic_divide(base, cascade_count, X, D, debug):
    if D == 0:
        raise ZeroDivisionError('Divisor cannot be zero.')
    [r_in, p_in, d_p, n, m] = normalize_input(base, X, D, debug)
    # return X // D
    for i in range(1, n - m + 2, cascade_count):
        for i_stage in range(i, i + cascade_count):
            [r_out, p_out] = divide_stage(base, r_in, p_in, d_p, n, m, i_stage, debug)
            r_in = r_out
            p_in = p_out
    quotient = adjust_result(r_in, p_in, d_p, debug)
    return quotient


def test_int_to_binary():
    min_int = -1000
    max_int = 1000
    for i in range(min_int,max_int+1):
        assert(i == signed_binary_to_integer(integer_to_signed_binary(i)))

def test_complement():
    min_int = -1000
    max_int = 1000
    for i in range(min_int,max_int+1):
        assert(-i == signed_binary_to_integer(complement_binary_number(integer_to_signed_binary(i))))



def test_add():
    min_a = -50
    max_a = 50
    min_b = -50
    max_b = 50
    for a in range(min_a,max_a+1):
        for b in range(min_b,max_b+1):
            bin_a = integer_to_signed_binary(a)
            bin_b = integer_to_signed_binary(b)
            bin_sum = signed_binary_to_integer(add_signed_binary_numbers(bin_a,bin_b))
            assert(a+b==bin_sum)

def test_mul():
    min_a = -50
    max_a = 50
    min_b = 0
    max_b = 50
    for a in range(min_a,max_a+1):
        for b in range(min_b,max_b+1):
            bin_a = integer_to_signed_binary(a)
            bin_b = integer_to_signed_binary(b)
            bin_sum = signed_binary_to_integer(carry_propagate(carry_save_multiply_binary_numbers(bin_a,bin_b)))
            assert(a*b==bin_sum)



def test(base, cascade_count):
    min_dvd = -1024
    max_dvd = 1024
    min_dvs = -1024
    max_dvs = 1024

    fail_count = 0
    pass_count = 0


    total_cases = (max_dvd - min_dvd + 1) * (max_dvs - min_dvs)
    perc_complete = 1
    num_complete  = 0

    for dividend in range(min_dvd, max_dvd + 1):
        for divisor in range(min_dvs, max_dvs + 1):
            if (divisor == 0):
                continue
            if abs(dividend // divisor - vedic_divide(base, cascade_count, dividend, divisor, False)) == 0:
                pass_count += 1
            else:
                print("WARN: Vedic Division failed for dividend: " + str(dividend) + " and divisor:" + str(divisor))
                fail_count += 1

            num_complete += 1
            if num_complete / total_cases * 100 >= perc_complete:
                print ("INFO: Simulation " + str(perc_complete) + "% Complete (" + str(num_complete) + "/" + str(total_cases) + ")" )
                perc_complete += 1


    print("Test Inputs:")
    print("############")
    print("Base: " + str(base))
    print("Cascade count: " + str(cascade_count))
    print("Min Dividend: " + str(min_dvd))
    print("Max Dividend: " + str(max_dvd))
    print("Min Divisor: " + str(min_dvs))
    print("Max Divisor: " + str(max_dvs))
    print("")
    print("Test Results:")
    print("#############")
    print("Pass Count: " + str(pass_count))
    print("Fail Count:" + str(fail_count))
    print("Pass %: " + str(pass_count * 100.0 / (pass_count + fail_count)))


test(8, 1)
#test_int_to_binary()
#test_add()
#test_mul()
#test_complement()

#print(-64//-15)
#vedic_divide(8,1,-64,-15,True)

