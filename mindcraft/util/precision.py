import struct
import numpy as np


def float32_to_binary(value):
    """Convert a float32 to its binary representation."""
    value = np.array(value, dtype="float32")
    [d] = struct.unpack('>I', struct.pack('>f', value))
    return f'{d:032b}'


def binary_to_float32(b):
    """Convert binary representation back to a float32."""
    bf = int(b, 2).to_bytes(4, byteorder='big')
    return struct.unpack('>f', bf)[0]


def reduce_precision_float32(value, precision_bits):
    """Reduce the precision of a float32 to a specified number of mantissa bits."""
    max_mantissa_bits = 23

    if precision_bits < 1 or precision_bits > max_mantissa_bits:
        raise ValueError(f"Precision bits should be between 1 and {max_mantissa_bits} for float32.")

    # Convert float32 to binary representation
    binary = float32_to_binary(value)

    # IEEE 754 single precision format:
    # 1 bit sign, 8 bits exponent, 23 bits mantissa
    sign_bit = binary[0]
    exponent_bits = binary[1:9]
    mantissa_bits = binary[9:]

    # Truncate the mantissa to the desired precision
    truncated_mantissa = mantissa_bits[:precision_bits] + '0' * (max_mantissa_bits - precision_bits)

    # Combine the parts back into a single binary string
    reduced_binary = sign_bit + exponent_bits + truncated_mantissa

    # Convert the truncated binary representation back to a float32
    reduced_precision_value = binary_to_float32(reduced_binary)

    return reduced_precision_value


def float16_to_binary(value):
    """Convert a float16 to its binary representation."""
    [d] = struct.unpack('>H', struct.pack('>e', value))
    return f'{d:016b}'


def binary_to_float16(b):
    """Convert binary representation back to a float16."""
    bf = int(b, 2).to_bytes(2, byteorder='big')
    return struct.unpack('>e', bf)[0]


def reduce_precision_float16(value, precision_bits):
    """Reduce the precision of a float16 to a specified number of mantissa bits."""
    max_mantissa_bits = 10

    if precision_bits < 1 or precision_bits > max_mantissa_bits:
        raise ValueError(f"Precision bits should be between 1 and {max_mantissa_bits} for float16.")

    # Convert float16 to binary representation
    binary = float16_to_binary(value)

    # IEEE 754 half precision format:
    # 1 bit sign, 5 bits exponent, 10 bits mantissa
    sign_bit = binary[0]
    exponent_bits = binary[1:6]
    mantissa_bits = binary[6:]

    # Truncate the mantissa to the desired precision
    truncated_mantissa = mantissa_bits[:precision_bits] + '0' * (max_mantissa_bits - precision_bits)

    # Combine the parts back into a single binary string
    reduced_binary = sign_bit + exponent_bits + truncated_mantissa

    # Convert the truncated binary representation back to a float16
    reduced_precision_value = binary_to_float16(reduced_binary)

    return reduced_precision_value
