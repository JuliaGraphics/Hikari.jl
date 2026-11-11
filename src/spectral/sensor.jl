# ============================================================================
# PixelSensor — Camera sensor spectral response model (pbrt-v4 compatible)
# ============================================================================
# Implements the sensor model from pbrt-v4: spectral response curves for real
# cameras (Nikon D850, Canon EOS 5D, Sony ILCE, etc.), ColorChecker-based
# calibration for XYZFromSensorRGB, white balance via Bradford adaptation,
# and ISO/exposure scaling.
#
# Reference: pbrt-v4 src/pbrt/film.h PixelSensor class

# ============================================================================
# Bradford chromatic adaptation matrices
# ============================================================================

const LMS_FROM_XYZ = Mat3f(
     0.8951f0,  -0.7502f0,   0.0389f0,
     0.2664f0,   1.7135f0,  -0.0685f0,
    -0.1614f0,   0.0367f0,   1.0296f0,
)

const XYZ_FROM_LMS = Mat3f(
     0.986993f0,   0.432305f0,  -0.00852866f0,
    -0.147054f0,   0.51836f0,    0.0400428f0,
     0.159963f0,   0.0492912f0,  0.968487f0,
)

# sRGB to XYZ and back (D65 white point)
const XYZ_FROM_SRGB = Mat3f(
    0.4124564f0, 0.2126729f0, 0.0193339f0,
    0.3575761f0, 0.7151522f0, 0.1191920f0,
    0.1804375f0, 0.0721750f0, 0.9503041f0,
)

const SRGB_FROM_XYZ = Mat3f(
     3.2404542f0, -0.9692660f0,  0.0556434f0,
    -1.5371385f0,  1.8760108f0, -0.2040259f0,
    -0.4985314f0,  0.0415560f0,  1.0572252f0,
)

# ============================================================================
# White balance via Bradford transform
# ============================================================================

"""
    white_balance(src_white_xy, target_white_xy) -> Mat3f

Compute a 3x3 chromatic adaptation matrix (Bradford) that maps
XYZ values lit by `src_white` illuminant to appear as if lit by `target_white`.
White points given as CIE xy chromaticity coordinates.
"""
function white_balance(src_xy::Point2f, target_xy::Point2f)
    # Convert xy → XYZ (Y=1)
    src_xyz = Vec3f(src_xy[1] / src_xy[2], 1f0, (1f0 - src_xy[1] - src_xy[2]) / src_xy[2])
    dst_xyz = Vec3f(target_xy[1] / target_xy[2], 1f0, (1f0 - target_xy[1] - target_xy[2]) / target_xy[2])

    # Convert to LMS
    src_lms = LMS_FROM_XYZ * src_xyz
    dst_lms = LMS_FROM_XYZ * dst_xyz

    # Diagonal scaling in LMS space
    lms_correct = Mat3f(
        dst_lms[1]/src_lms[1], 0f0, 0f0,
        0f0, dst_lms[2]/src_lms[2], 0f0,
        0f0, 0f0, dst_lms[3]/src_lms[3],
    )

    return XYZ_FROM_LMS * lms_correct * LMS_FROM_XYZ
end

# D65 white point in CIE xy
const D65_WHITE_XY = Point2f(0.3127f0, 0.3290f0)

# ============================================================================
# CIE D illuminant from color temperature
# ============================================================================

"""
    cie_d_illuminant_xy(temperature_K) -> Point2f

Compute the CIE xy chromaticity of a D-series illuminant at given temperature.
Valid for 4000K-25000K.
"""
function cie_d_illuminant_xy(T::Float32)
    # CIE formula for D illuminant chromaticity
    T_inv = 1f0 / T
    xD = if T <= 7000f0
        0.244063f0 + 0.09911f0 * 1000f0 * T_inv + 2.9678f0 * 1f6 * T_inv^2 - 4.6070f0 * 1f9 * T_inv^3
    else
        0.237040f0 + 0.24748f0 * 1000f0 * T_inv + 1.9018f0 * 1f6 * T_inv^2 - 2.0064f0 * 1f9 * T_inv^3
    end
    yD = -3f0 * xD^2 + 2.87f0 * xD - 0.275f0
    return Point2f(xD, yD)
end

# ============================================================================
# Sensor spectral response curves
# ============================================================================

struct SensorCurves{N}
    r::PiecewiseLinearSpectrum{N}
    g::PiecewiseLinearSpectrum{N}
    b::PiecewiseLinearSpectrum{N}
end

# Nikon D850 spectral response (from pbrt-v4 spectrum.cpp)
const NIKON_D850_R = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0, 0.001324, 390.0, 0.001665, 400.0, 0.001879, 410.0, 0.001631,
    420.0, 0.005657, 430.0, 0.008393, 440.0, 0.004228, 450.0, 0.002865,
    460.0, 0.002569, 470.0, 0.005162, 480.0, 0.008174, 490.0, 0.010304,
    500.0, 0.013883, 510.0, 0.017953, 520.0, 0.035428, 530.0, 0.033022,
    540.0, 0.031189, 550.0, 0.031901, 560.0, 0.039088, 570.0, 0.105125,
    580.0, 0.379337, 590.0, 0.675574, 600.0, 0.646691, 610.0, 0.584900,
    620.0, 0.514731, 630.0, 0.394011, 640.0, 0.328469, 650.0, 0.214064,
    660.0, 0.161475, 670.0, 0.116392, 680.0, 0.036028, 690.0, 0.007949,
    700.0, 0.003235, 710.0, 0.002354, 720.0, 0.001518,
))

const NIKON_D850_G = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0, 0.001234, 390.0, 0.000932, 400.0, 0.001010, 410.0, 0.000897,
    420.0, 0.004591, 430.0, 0.009230, 440.0, 0.016991, 450.0, 0.020811,
    460.0, 0.025510, 470.0, 0.156180, 480.0, 0.421466, 490.0, 0.600582,
    500.0, 0.735533, 510.0, 0.904917, 520.0, 1.000000, 530.0, 0.827569,
    540.0, 0.768431, 550.0, 0.737256, 560.0, 0.703231, 570.0, 0.555365,
    580.0, 0.443409, 590.0, 0.319246, 600.0, 0.180470, 610.0, 0.088453,
    620.0, 0.039125, 630.0, 0.021046, 640.0, 0.013167, 650.0, 0.007409,
    660.0, 0.005700, 670.0, 0.005477, 680.0, 0.002611, 690.0, 0.001025,
    700.0, 0.000759, 710.0, 0.000587, 720.0, 0.001518,
))

const NIKON_D850_B = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0, 0.002306, 390.0, 0.004650, 400.0, 0.011389, 410.0, 0.018650,
    420.0, 0.173800, 430.0, 0.491892, 440.0, 0.624891, 450.0, 0.745966,
    460.0, 0.733487, 470.0, 0.816500, 480.0, 0.836138, 490.0, 0.657176,
    500.0, 0.538470, 510.0, 0.321903, 520.0, 0.278352, 530.0, 0.148652,
    540.0, 0.091618, 550.0, 0.052584, 560.0, 0.030906, 570.0, 0.016207,
    580.0, 0.010997, 590.0, 0.008064, 600.0, 0.004920, 610.0, 0.003378,
    620.0, 0.003120, 630.0, 0.002948, 640.0, 0.003768, 650.0, 0.004110,
    660.0, 0.004624, 670.0, 0.004326, 680.0, 0.001633, 690.0, 0.000457,
    700.0, 0.000284, 710.0, 0.000215, 720.0, 0.001518,
))

const NIKON_D850 = SensorCurves(NIKON_D850_R, NIKON_D850_G, NIKON_D850_B)

# Registry of available sensors
const SENSOR_REGISTRY = Dict{String, SensorCurves{35}}(
    "nikon_d850" => NIKON_D850,
)

# ============================================================================
# ColorChecker swatch reflectances (BabelColor data, 24 patches)
# ============================================================================

const COLORCHECKER_SWATCHES = [
    # 1: Dark Skin
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.055,390.0,0.058,400.0,0.061,410.0,0.062,420.0,0.062,430.0,0.062,440.0,0.062,450.0,0.062,460.0,0.062,470.0,0.062,480.0,0.062,490.0,0.063,500.0,0.065,510.0,0.070,520.0,0.076,530.0,0.079,540.0,0.081,550.0,0.084,560.0,0.091,570.0,0.103,580.0,0.119,590.0,0.134,600.0,0.143,610.0,0.147,620.0,0.151,630.0,0.158,640.0,0.168,650.0,0.179,660.0,0.188,670.0,0.190,680.0,0.186,690.0,0.181,700.0,0.182,710.0,0.187,720.0,0.196,730.0,0.209)),
    # 2: Light Skin
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.117,390.0,0.143,400.0,0.175,410.0,0.191,420.0,0.196,430.0,0.199,440.0,0.204,450.0,0.213,460.0,0.228,470.0,0.251,480.0,0.280,490.0,0.309,500.0,0.329,510.0,0.333,520.0,0.315,530.0,0.286,540.0,0.273,550.0,0.276,560.0,0.277,570.0,0.289,580.0,0.339,590.0,0.420,600.0,0.488,610.0,0.525,620.0,0.546,630.0,0.562,640.0,0.578,650.0,0.595,660.0,0.612,670.0,0.625,680.0,0.638,690.0,0.656,700.0,0.678,710.0,0.700,720.0,0.717,730.0,0.734)),
    # 3: Blue Flower
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.130,390.0,0.177,400.0,0.251,410.0,0.306,420.0,0.324,430.0,0.330,440.0,0.333,450.0,0.331,460.0,0.323,470.0,0.311,480.0,0.298,490.0,0.285,500.0,0.269,510.0,0.250,520.0,0.231,530.0,0.214,540.0,0.199,550.0,0.185,560.0,0.169,570.0,0.157,580.0,0.149,590.0,0.145,600.0,0.142,610.0,0.141,620.0,0.141,630.0,0.141,640.0,0.143,650.0,0.147,660.0,0.152,670.0,0.154,680.0,0.150,690.0,0.144,700.0,0.136,710.0,0.132,720.0,0.135,730.0,0.147)),
    # 4: Brown
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.051,390.0,0.054,400.0,0.056,410.0,0.057,420.0,0.058,430.0,0.059,440.0,0.060,450.0,0.061,460.0,0.062,470.0,0.063,480.0,0.065,490.0,0.067,500.0,0.075,510.0,0.101,520.0,0.145,530.0,0.178,540.0,0.184,550.0,0.170,560.0,0.149,570.0,0.133,580.0,0.122,590.0,0.115,600.0,0.109,610.0,0.105,620.0,0.104,630.0,0.106,640.0,0.109,650.0,0.112,660.0,0.114,670.0,0.114,680.0,0.112,690.0,0.112,700.0,0.115,710.0,0.120,720.0,0.125,730.0,0.130)),
    # 5: Cyan
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.144,390.0,0.198,400.0,0.294,410.0,0.375,420.0,0.408,430.0,0.421,440.0,0.426,450.0,0.426,460.0,0.419,470.0,0.403,480.0,0.379,490.0,0.346,500.0,0.311,510.0,0.281,520.0,0.254,530.0,0.229,540.0,0.214,550.0,0.208,560.0,0.202,570.0,0.194,580.0,0.193,590.0,0.200,600.0,0.214,610.0,0.230,620.0,0.241,630.0,0.254,640.0,0.279,650.0,0.313,660.0,0.348,670.0,0.366,680.0,0.366,690.0,0.359,700.0,0.358,710.0,0.365,720.0,0.377,730.0,0.398)),
    # 6: Green
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.136,390.0,0.179,400.0,0.247,410.0,0.297,420.0,0.320,430.0,0.337,440.0,0.355,450.0,0.381,460.0,0.419,470.0,0.466,480.0,0.510,490.0,0.546,500.0,0.567,510.0,0.574,520.0,0.569,530.0,0.551,540.0,0.524,550.0,0.488,560.0,0.445,570.0,0.400,580.0,0.350,590.0,0.299,600.0,0.252,610.0,0.221,620.0,0.204,630.0,0.196,640.0,0.191,650.0,0.188,660.0,0.191,670.0,0.199,680.0,0.212,690.0,0.223,700.0,0.232,710.0,0.233,720.0,0.229,730.0,0.229)),
    # 7: Red
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.054,390.0,0.054,400.0,0.053,410.0,0.054,420.0,0.054,430.0,0.055,440.0,0.055,450.0,0.055,460.0,0.056,470.0,0.057,480.0,0.058,490.0,0.061,500.0,0.068,510.0,0.089,520.0,0.125,530.0,0.154,540.0,0.174,550.0,0.199,560.0,0.248,570.0,0.335,580.0,0.444,590.0,0.538,600.0,0.587,610.0,0.595,620.0,0.591,630.0,0.587,640.0,0.584,650.0,0.584,660.0,0.590,670.0,0.603,680.0,0.620,690.0,0.639,700.0,0.655,710.0,0.663,720.0,0.663,730.0,0.667)),
    # 8: Yellow
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.122,390.0,0.164,400.0,0.229,410.0,0.286,420.0,0.327,430.0,0.361,440.0,0.388,450.0,0.400,460.0,0.392,470.0,0.362,480.0,0.316,490.0,0.260,500.0,0.209,510.0,0.168,520.0,0.138,530.0,0.117,540.0,0.104,550.0,0.096,560.0,0.090,570.0,0.086,580.0,0.084,590.0,0.084,600.0,0.084,610.0,0.084,620.0,0.084,630.0,0.085,640.0,0.090,650.0,0.098,660.0,0.109,670.0,0.123,680.0,0.143,690.0,0.169,700.0,0.205,710.0,0.244,720.0,0.287,730.0,0.332)),
    # 9: Magenta
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.096,390.0,0.115,400.0,0.131,410.0,0.135,420.0,0.133,430.0,0.132,440.0,0.130,450.0,0.128,460.0,0.125,470.0,0.120,480.0,0.115,490.0,0.110,500.0,0.105,510.0,0.100,520.0,0.095,530.0,0.093,540.0,0.092,550.0,0.093,560.0,0.096,570.0,0.108,580.0,0.156,590.0,0.265,600.0,0.399,610.0,0.500,620.0,0.556,630.0,0.579,640.0,0.588,650.0,0.591,660.0,0.593,670.0,0.594,680.0,0.598,690.0,0.602,700.0,0.607,710.0,0.609,720.0,0.609,730.0,0.610)),
    # 10: Blue
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.092,390.0,0.116,400.0,0.146,410.0,0.169,420.0,0.178,430.0,0.173,440.0,0.158,450.0,0.139,460.0,0.119,470.0,0.101,480.0,0.087,490.0,0.075,500.0,0.066,510.0,0.060,520.0,0.056,530.0,0.053,540.0,0.051,550.0,0.051,560.0,0.052,570.0,0.052,580.0,0.051,590.0,0.052,600.0,0.058,610.0,0.073,620.0,0.096,630.0,0.119,640.0,0.141,650.0,0.166,660.0,0.194,670.0,0.227,680.0,0.265,690.0,0.309,700.0,0.355,710.0,0.396,720.0,0.436,730.0,0.478)),
    # 11: Green-Yellow
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.061,390.0,0.061,400.0,0.062,410.0,0.063,420.0,0.064,430.0,0.066,440.0,0.069,450.0,0.075,460.0,0.085,470.0,0.105,480.0,0.139,490.0,0.192,500.0,0.271,510.0,0.376,520.0,0.476,530.0,0.531,540.0,0.549,550.0,0.546,560.0,0.528,570.0,0.504,580.0,0.471,590.0,0.428,600.0,0.381,610.0,0.347,620.0,0.327,630.0,0.318,640.0,0.312,650.0,0.310,660.0,0.314,670.0,0.327,680.0,0.345,690.0,0.363,700.0,0.376,710.0,0.381,720.0,0.378,730.0,0.379)),
    # 12: Orange
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.063,390.0,0.063,400.0,0.063,410.0,0.064,420.0,0.064,430.0,0.064,440.0,0.065,450.0,0.066,460.0,0.067,470.0,0.068,480.0,0.071,490.0,0.076,500.0,0.087,510.0,0.125,520.0,0.206,530.0,0.305,540.0,0.383,550.0,0.431,560.0,0.469,570.0,0.518,580.0,0.568,590.0,0.607,600.0,0.628,610.0,0.637,620.0,0.640,630.0,0.642,640.0,0.645,650.0,0.648,660.0,0.651,670.0,0.653,680.0,0.657,690.0,0.664,700.0,0.673,710.0,0.680,720.0,0.684,730.0,0.688)),
    # 13: Red-Blue
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.066,390.0,0.079,400.0,0.102,410.0,0.146,420.0,0.200,430.0,0.244,440.0,0.282,450.0,0.309,460.0,0.308,470.0,0.278,480.0,0.231,490.0,0.178,500.0,0.130,510.0,0.094,520.0,0.070,530.0,0.054,540.0,0.046,550.0,0.042,560.0,0.039,570.0,0.038,580.0,0.038,590.0,0.038,600.0,0.038,610.0,0.039,620.0,0.039,630.0,0.040,640.0,0.041,650.0,0.042,660.0,0.044,670.0,0.045,680.0,0.046,690.0,0.046,700.0,0.048,710.0,0.052,720.0,0.057,730.0,0.065)),
    # 14: Light Blue
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.052,390.0,0.053,400.0,0.054,410.0,0.055,420.0,0.057,430.0,0.059,440.0,0.061,450.0,0.066,460.0,0.075,470.0,0.093,480.0,0.125,490.0,0.178,500.0,0.246,510.0,0.307,520.0,0.337,530.0,0.334,540.0,0.317,550.0,0.293,560.0,0.262,570.0,0.230,580.0,0.198,590.0,0.165,600.0,0.135,610.0,0.115,620.0,0.104,630.0,0.098,640.0,0.094,650.0,0.092,660.0,0.093,670.0,0.097,680.0,0.102,690.0,0.108,700.0,0.113,710.0,0.115,720.0,0.114,730.0,0.114)),
    # 15: Black (Red)
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.050,390.0,0.049,400.0,0.048,410.0,0.047,420.0,0.047,430.0,0.047,440.0,0.047,450.0,0.047,460.0,0.046,470.0,0.045,480.0,0.044,490.0,0.044,500.0,0.045,510.0,0.046,520.0,0.047,530.0,0.048,540.0,0.049,550.0,0.050,560.0,0.054,570.0,0.060,580.0,0.072,590.0,0.104,600.0,0.178,610.0,0.312,620.0,0.467,630.0,0.581,640.0,0.644,650.0,0.675,660.0,0.690,670.0,0.698,680.0,0.706,690.0,0.715,700.0,0.724,710.0,0.730,720.0,0.734,730.0,0.738)),
    # 16: Cyan-Green
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.058,390.0,0.054,400.0,0.052,410.0,0.052,420.0,0.053,430.0,0.054,440.0,0.056,450.0,0.059,460.0,0.067,470.0,0.081,480.0,0.107,490.0,0.152,500.0,0.225,510.0,0.336,520.0,0.462,530.0,0.559,540.0,0.616,550.0,0.650,560.0,0.672,570.0,0.694,580.0,0.710,590.0,0.723,600.0,0.731,610.0,0.739,620.0,0.746,630.0,0.752,640.0,0.758,650.0,0.764,660.0,0.769,670.0,0.771,680.0,0.776,690.0,0.782,700.0,0.790,710.0,0.796,720.0,0.799,730.0,0.804)),
    # 17: Purple
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.145,390.0,0.195,400.0,0.283,410.0,0.346,420.0,0.362,430.0,0.354,440.0,0.334,450.0,0.306,460.0,0.276,470.0,0.248,480.0,0.218,490.0,0.190,500.0,0.168,510.0,0.149,520.0,0.127,530.0,0.107,540.0,0.100,550.0,0.102,560.0,0.104,570.0,0.109,580.0,0.137,590.0,0.200,600.0,0.290,610.0,0.400,620.0,0.516,630.0,0.615,640.0,0.687,650.0,0.732,660.0,0.760,670.0,0.774,680.0,0.783,690.0,0.793,700.0,0.803,710.0,0.812,720.0,0.817,730.0,0.825)),
    # 18: Blue Sky
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.108,390.0,0.141,400.0,0.192,410.0,0.236,420.0,0.261,430.0,0.286,440.0,0.317,450.0,0.353,460.0,0.390,470.0,0.426,480.0,0.446,490.0,0.444,500.0,0.423,510.0,0.385,520.0,0.337,530.0,0.283,540.0,0.231,550.0,0.185,560.0,0.146,570.0,0.118,580.0,0.101,590.0,0.090,600.0,0.082,610.0,0.076,620.0,0.074,630.0,0.073,640.0,0.073,650.0,0.074,660.0,0.076,670.0,0.077,680.0,0.076,690.0,0.075,700.0,0.073,710.0,0.072,720.0,0.074,730.0,0.079)),
    # 19: White
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.189,390.0,0.255,400.0,0.423,410.0,0.660,420.0,0.811,430.0,0.862,440.0,0.877,450.0,0.884,460.0,0.891,470.0,0.896,480.0,0.899,490.0,0.904,500.0,0.907,510.0,0.909,520.0,0.911,530.0,0.910,540.0,0.911,550.0,0.914,560.0,0.913,570.0,0.916,580.0,0.915,590.0,0.916,600.0,0.914,610.0,0.915,620.0,0.918,630.0,0.919,640.0,0.921,650.0,0.923,660.0,0.924,670.0,0.922,680.0,0.922,690.0,0.925,700.0,0.927,710.0,0.930,720.0,0.930,730.0,0.933)),
    # 20: Light Grey
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.171,390.0,0.232,400.0,0.365,410.0,0.507,420.0,0.567,430.0,0.583,440.0,0.588,450.0,0.590,460.0,0.591,470.0,0.590,480.0,0.588,490.0,0.588,500.0,0.589,510.0,0.589,520.0,0.591,530.0,0.590,540.0,0.590,550.0,0.590,560.0,0.589,570.0,0.591,580.0,0.590,590.0,0.590,600.0,0.587,610.0,0.585,620.0,0.583,630.0,0.580,640.0,0.578,650.0,0.576,660.0,0.574,670.0,0.572,680.0,0.571,690.0,0.569,700.0,0.568,710.0,0.568,720.0,0.566,730.0,0.566)),
    # 21: Medium Grey
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.144,390.0,0.192,400.0,0.272,410.0,0.331,420.0,0.350,430.0,0.357,440.0,0.361,450.0,0.363,460.0,0.363,470.0,0.361,480.0,0.359,490.0,0.358,500.0,0.358,510.0,0.359,520.0,0.360,530.0,0.360,540.0,0.361,550.0,0.361,560.0,0.360,570.0,0.362,580.0,0.362,590.0,0.361,600.0,0.359,610.0,0.358,620.0,0.355,630.0,0.352,640.0,0.350,650.0,0.348,660.0,0.345,670.0,0.343,680.0,0.340,690.0,0.338,700.0,0.335,710.0,0.334,720.0,0.332,730.0,0.331)),
    # 22: Dark Grey
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.105,390.0,0.131,400.0,0.163,410.0,0.180,420.0,0.186,430.0,0.190,440.0,0.193,450.0,0.194,460.0,0.194,470.0,0.192,480.0,0.191,490.0,0.191,500.0,0.191,510.0,0.192,520.0,0.192,530.0,0.192,540.0,0.192,550.0,0.192,560.0,0.192,570.0,0.193,580.0,0.192,590.0,0.192,600.0,0.191,610.0,0.189,620.0,0.188,630.0,0.186,640.0,0.184,650.0,0.182,660.0,0.181,670.0,0.179,680.0,0.178,690.0,0.176,700.0,0.174,710.0,0.173,720.0,0.172,730.0,0.171)),
    # 23: Very Dark Grey
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.068,390.0,0.077,400.0,0.084,410.0,0.087,420.0,0.089,430.0,0.090,440.0,0.092,450.0,0.092,460.0,0.091,470.0,0.090,480.0,0.090,490.0,0.090,500.0,0.090,510.0,0.090,520.0,0.090,530.0,0.090,540.0,0.090,550.0,0.090,560.0,0.090,570.0,0.090,580.0,0.090,590.0,0.089,600.0,0.089,610.0,0.088,620.0,0.087,630.0,0.086,640.0,0.086,650.0,0.085,660.0,0.084,670.0,0.084,680.0,0.083,690.0,0.083,700.0,0.082,710.0,0.081,720.0,0.081,730.0,0.081)),
    # 24: Black
    from_interleaved(PiecewiseLinearSpectrum{36}, (380.0,0.031,390.0,0.032,400.0,0.032,410.0,0.033,420.0,0.033,430.0,0.033,440.0,0.033,450.0,0.033,460.0,0.032,470.0,0.032,480.0,0.032,490.0,0.032,500.0,0.032,510.0,0.032,520.0,0.032,530.0,0.032,540.0,0.032,550.0,0.032,560.0,0.032,570.0,0.032,580.0,0.032,590.0,0.032,600.0,0.032,610.0,0.032,620.0,0.032,630.0,0.032,640.0,0.032,650.0,0.032,660.0,0.032,670.0,0.032,680.0,0.032,690.0,0.032,700.0,0.032,710.0,0.032,720.0,0.032,730.0,0.033)),
]

# ============================================================================
# PixelSensor type
# ============================================================================

"""
    PixelSensor

Camera sensor model matching pbrt-v4's PixelSensor. Converts spectral radiance
to output RGB via sensor response curves, ColorChecker calibration, white balance,
and ISO/exposure scaling.

# Fields
- `output_from_sensor`: 3x3 matrix = RGBFromXYZ * XYZFromSensorRGB
- `imaging_ratio`: exposure_time * ISO / 100
"""
struct PixelSensor
    output_from_sensor::Mat3f  # combines sensor→XYZ→sRGB
    imaging_ratio::Float32
    sensor_name::String
end

"""
    PixelSensor(; sensor="cie1931", iso=100, whitebalance=0, exposure_time=1.0)

Create a pixel sensor matching pbrt-v4's defaults.
"""
function PixelSensor(; sensor::String="cie1931",
                     iso=100f0,
                     whitebalance=0f0,
                     exposure_time=1f0)
    imaging_ratio = Float32(exposure_time) * Float32(iso) / 100f0
    _whitebalance = Float32(whitebalance)
    if sensor == "cie1931"
        # CIE XYZ matching functions → just white balance
        if _whitebalance == 0f0
            # No white balance: identity mapping XYZ→XYZ, then sRGB
            output_from_sensor = SRGB_FROM_XYZ
        else
            wb_xy = cie_d_illuminant_xy(_whitebalance)
            wb_matrix = white_balance(wb_xy, D65_WHITE_XY)
            output_from_sensor = SRGB_FROM_XYZ * wb_matrix
        end
        return PixelSensor(output_from_sensor, imaging_ratio, sensor)
    end

    # Real camera sensor
    sensor_key = lowercase(sensor)
    haskey(SENSOR_REGISTRY, sensor_key) || error("Unknown sensor: $sensor. Available: $(join(keys(SENSOR_REGISTRY), ", "))")
    curves = SENSOR_REGISTRY[sensor_key]

    # Determine white balance illuminant
    wb_temp = _whitebalance == 0f0 ? 6500f0 : _whitebalance
    wb_xy = cie_d_illuminant_xy(wb_temp)

    # Compute XYZFromSensorRGB via ColorChecker calibration
    xyz_from_sensor = calibrate_sensor(curves, wb_temp)

    # Output transform: sRGB from XYZ * XYZ from sensor
    output_from_sensor = SRGB_FROM_XYZ * xyz_from_sensor

    return PixelSensor(output_from_sensor, imaging_ratio, sensor_key)
end

# ============================================================================
# ColorChecker-based sensor calibration
# ============================================================================

"""
Evaluate a PiecewiseLinearSpectrum at a given wavelength via linear interpolation.
"""
function evaluate_pls(spec::PiecewiseLinearSpectrum{N}, lambda::Float32) where N
    sample(spec, lambda)
end

"""
Integrate spectrum * illuminant * response over visible range using simple trapezoid rule.
Returns the integral from 380nm to 720nm.
"""
function spectral_integral(response, illuminant, lambda_start=380f0, lambda_end=720f0, n_steps=68)
    h = (lambda_end - lambda_start) / n_steps
    total = 0f0
    for i in 0:n_steps
        lambda = lambda_start + i * h
        w = (i == 0 || i == n_steps) ? 0.5f0 : 1f0
        total += w * evaluate_pls(response, lambda) * evaluate_pls(illuminant, lambda)
    end
    return total * h
end

"""
Integrate reflectance * illuminant * response over visible range, normalized by
∫ illuminant * g_response dλ (matching pbrt-v4's ProjectReflectance).
"""
function project_reflectance(reflectance::PiecewiseLinearSpectrum,
                             illuminant::PiecewiseLinearSpectrum,
                             r_bar, g_bar, b_bar)
    r = spectral_integral_3(reflectance, illuminant, r_bar)
    g = spectral_integral_3(reflectance, illuminant, g_bar)
    b = spectral_integral_3(reflectance, illuminant, b_bar)
    g_norm = spectral_integral(g_bar, illuminant)
    return Vec3f(r / g_norm, g / g_norm, b / g_norm)
end

function spectral_integral_3(reflectance, illuminant, response,
                             lambda_start=380f0, lambda_end=720f0, n_steps=68)
    h = (lambda_end - lambda_start) / n_steps
    total = 0f0
    for i in 0:n_steps
        lambda = lambda_start + i * h
        w = (i == 0 || i == n_steps) ? 0.5f0 : 1f0
        total += w * evaluate_pls(reflectance, lambda) * evaluate_pls(illuminant, lambda) * evaluate_pls(response, lambda)
    end
    return total * h
end

"""
Calibrate sensor using ColorChecker swatches via linear least squares.
Returns a 3x3 XYZFromSensorRGB matrix.
"""
function calibrate_sensor(curves::SensorCurves{N}, wb_temp::Float32) where N
    # Sensor illuminant: D(wb_temp) — matches pbrt-v4's PixelSensor::Create
    sensor_illum = cie_d_illuminant_spectrum(wb_temp)

    # CIE matching functions
    cie_x = CIE_X_TABLE
    cie_y = CIE_Y_TABLE
    cie_z = CIE_Z_TABLE

    # sRGB output color space illuminant = D65
    output_illum = D65_ILLUMINANT_TABLE

    # Compute sensor white G normalization
    sensor_white_g = spectral_integral(curves.g, sensor_illum)
    sensor_white_y = spectral_integral(cie_y, sensor_illum)

    n_swatches = length(COLORCHECKER_SWATCHES)
    # A = sensor RGB, B = output XYZ (both N×3)
    A = zeros(Float32, n_swatches, 3)
    B = zeros(Float32, n_swatches, 3)

    for i in 1:n_swatches
        swatch = COLORCHECKER_SWATCHES[i]
        # Sensor RGB
        rgb = project_reflectance(swatch, sensor_illum, curves.r, curves.g, curves.b)
        A[i, 1] = rgb[1]; A[i, 2] = rgb[2]; A[i, 3] = rgb[3]

        # Output XYZ
        xyz = project_reflectance(swatch, output_illum, cie_x, cie_y, cie_z)
        # Scale by sensor_white_Y / sensor_white_G
        scale = sensor_white_y / sensor_white_g
        B[i, 1] = xyz[1] * scale; B[i, 2] = xyz[2] * scale; B[i, 3] = xyz[3] * scale
    end

    # Linear least squares: (A'A)\(A'B) gives M where xyz = sensor * M (right-multiply)
    # Transpose to get M' where xyz = M' * sensor (left-multiply, for Mat3f * Vec3f)
    M = ((A' * A) \ (A' * B))'

    # Column-major Mat3f
    return Mat3f(
        M[1,1], M[2,1], M[3,1],
        M[1,2], M[2,2], M[3,2],
        M[1,3], M[2,3], M[3,3],
    )
end

# ============================================================================
# CIE and D65 data as PiecewiseLinearSpectrum for calibration
# ============================================================================

# These are sampled versions for the calibration integral — not the GPU tables
const CIE_X_TABLE = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0,0.0014,390.0,0.0042,400.0,0.0143,410.0,0.0435,420.0,0.1344,430.0,0.2839,
    440.0,0.3483,450.0,0.3362,460.0,0.2908,470.0,0.1954,480.0,0.0956,490.0,0.032,
    500.0,0.0049,510.0,0.0093,520.0,0.0633,530.0,0.1655,540.0,0.2904,550.0,0.4334,
    560.0,0.5945,570.0,0.7621,580.0,0.9163,590.0,1.0263,600.0,1.0622,610.0,1.0026,
    620.0,0.8544,630.0,0.6424,640.0,0.4479,650.0,0.2835,660.0,0.1649,670.0,0.0874,
    680.0,0.0468,690.0,0.0227,700.0,0.0114,710.0,0.0058,720.0,0.0029,
))

const CIE_Y_TABLE = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0,0.0,390.0,0.0001,400.0,0.0004,410.0,0.0012,420.0,0.004,430.0,0.0116,
    440.0,0.023,450.0,0.038,460.0,0.06,470.0,0.091,480.0,0.139,490.0,0.208,
    500.0,0.323,510.0,0.503,520.0,0.71,530.0,0.862,540.0,0.954,550.0,0.995,
    560.0,0.995,570.0,0.952,580.0,0.87,590.0,0.757,600.0,0.631,610.0,0.503,
    620.0,0.381,630.0,0.265,640.0,0.175,650.0,0.107,660.0,0.061,670.0,0.032,
    680.0,0.017,690.0,0.0082,700.0,0.0041,710.0,0.0021,720.0,0.001,
))

const CIE_Z_TABLE = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0,0.0065,390.0,0.0201,400.0,0.0679,410.0,0.2074,420.0,0.6456,430.0,1.3856,
    440.0,1.7471,450.0,1.7721,460.0,1.6692,470.0,1.2876,480.0,0.813,490.0,0.4652,
    500.0,0.272,510.0,0.1582,520.0,0.0782,530.0,0.0422,540.0,0.0203,550.0,0.0087,
    560.0,0.0039,570.0,0.0021,580.0,0.0017,590.0,0.0011,600.0,0.0008,610.0,0.0003,
    620.0,0.0002,630.0,0.0,640.0,0.0,650.0,0.0,660.0,0.0,670.0,0.0,
    680.0,0.0,690.0,0.0,700.0,0.0,710.0,0.0,720.0,0.0,
))

"""
    sensor_response_table(sensor::PixelSensor) -> CIEXYZTable

Create a CIEXYZTable where the X/Y/Z channels contain sensor R/G/B response
curves (or CIE XYZ for the default cie1931 sensor). Resampled to 1nm grid
matching the CIE table format.
"""
function sensor_response_table(sensor_name::String)
    sensor_name = lowercase(sensor_name)
    if sensor_name == "cie1931"
        return CIEXYZTable()  # default CIE XYZ
    end

    haskey(SENSOR_REGISTRY, sensor_name) || error("Unknown sensor: $sensor_name")
    curves = SENSOR_REGISTRY[sensor_name]

    # Resample sensor curves to 1nm grid (CIE_LAMBDA_MIN:CIE_LAMBDA_MAX)
    r_data = Vector{Float32}(undef, N_CIE_SAMPLES)
    g_data = Vector{Float32}(undef, N_CIE_SAMPLES)
    b_data = Vector{Float32}(undef, N_CIE_SAMPLES)

    for i in 1:N_CIE_SAMPLES
        lambda = Float32(CIE_LAMBDA_MIN + i - 1)
        r_data[i] = sample(curves.r, lambda)
        g_data[i] = sample(curves.g, lambda)
        b_data[i] = sample(curves.b, lambda)
    end

    return CIEXYZTable(r_data, g_data, b_data)
end

# ============================================================================
# CIE D illuminant generation from S0/S1/S2 basis vectors
# ============================================================================

# CIE S basis vectors (107 points, 300-830nm at 5nm steps)
const CIE_S_LAMBDA = Float32[300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,500,505,510,515,520,525,530,535,540,545,550,555,560,565,570,575,580,585,590,595,600,605,610,615,620,625,630,635,640,645,650,655,660,665,670,675,680,685,690,695,700,705,710,715,720,725,730,735,740,745,750,755,760,765,770,775,780,785,790,795,800,805,810,815,820,825,830]

const CIE_S0 = Float32[0.04,3.02,6.0,17.8,29.6,42.45,55.3,56.3,57.3,59.55,61.8,61.65,61.5,65.15,68.8,66.1,63.4,64.6,65.8,80.3,94.8,99.8,104.8,105.35,105.9,101.35,96.8,105.35,113.9,119.75,125.6,125.55,125.5,123.4,121.3,121.3,121.3,117.4,113.5,113.3,113.1,111.95,110.8,108.65,106.5,107.65,108.8,107.05,105.3,104.85,104.4,102.2,100.0,98.0,96.0,95.55,95.1,92.1,89.1,89.8,90.5,90.4,90.3,89.35,88.4,86.2,84.0,84.55,85.1,83.5,81.9,82.25,82.6,83.75,84.9,83.1,81.3,76.6,71.9,73.1,74.3,75.35,76.4,69.85,63.3,67.5,71.7,74.35,77.0,71.1,65.2,56.45,47.7,58.15,68.6,66.8,65.0,65.5,66.0,63.5,61.0,57.15,53.3,56.1,58.9,60.4,61.9]

const CIE_S1 = Float32[0.02,2.26,4.5,13.45,22.4,32.2,42.0,41.3,40.6,41.1,41.6,39.8,38.0,40.2,42.4,40.45,38.5,36.75,35.0,39.2,43.4,44.85,46.3,45.1,43.9,40.5,37.1,36.9,36.7,36.3,35.9,34.25,32.6,30.25,27.9,26.1,24.3,22.2,20.1,18.15,16.2,14.7,13.2,10.9,8.6,7.35,6.1,5.15,4.2,3.05,1.9,0.95,0.0,-0.8,-1.6,-2.55,-3.5,-3.5,-3.5,-4.65,-5.8,-6.5,-7.2,-7.9,-8.6,-9.05,-9.5,-10.2,-10.9,-10.8,-10.7,-11.35,-12.0,-13.0,-14.0,-13.8,-13.6,-12.8,-12.0,-12.65,-13.3,-13.1,-12.9,-11.75,-10.6,-11.1,-11.6,-11.9,-12.2,-11.2,-10.2,-9.0,-7.8,-9.5,-11.2,-10.8,-10.4,-10.5,-10.6,-10.15,-9.7,-9.0,-8.3,-8.8,-9.3,-9.55,-9.8]

const CIE_S2 = Float32[0.0,1.0,2.0,3.0,4.0,6.25,8.5,8.15,7.8,7.25,6.7,6.0,5.3,5.7,6.1,4.55,3.0,2.1,1.2,0.05,-1.1,-0.8,-0.5,-0.6,-0.7,-0.95,-1.2,-1.9,-2.6,-2.75,-2.9,-2.85,-2.8,-2.7,-2.6,-2.6,-2.6,-2.2,-1.8,-1.65,-1.5,-1.4,-1.3,-1.25,-1.2,-1.1,-1.0,-0.75,-0.5,-0.4,-0.3,-0.15,0.0,0.1,0.2,0.35,0.5,1.3,2.1,2.65,3.2,3.65,4.1,4.4,4.7,4.9,5.1,5.9,6.7,7.0,7.3,7.95,8.6,9.2,9.8,10.0,10.2,9.25,8.3,8.95,9.6,9.05,8.5,7.75,7.0,7.3,7.6,7.8,8.0,7.35,6.7,5.95,5.2,6.3,7.4,7.1,6.8,6.9,7.0,6.7,6.4,5.95,5.5,5.8,6.1,6.3,6.5]

"""
    planck_blackbody_spectrum(T_kelvin) -> PiecewiseLinearSpectrum{107}

Generate a Planck blackbody SPD at the given temperature, sampled at the same
wavelength grid as the CIE S basis (300-830nm, 5nm steps). Used as fallback
for CIE D illuminant below 4000K. Matches pbrt-v4's behavior.
"""
function planck_blackbody_spectrum(T::Float32)
    # Planck's law: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    # Constants: h*c = 1.98645e-25 J·m, k = 1.38065e-23 J/K
    # λ in meters, output in W/sr/m²/m
    c1 = 3.7417749f14  # 2π h c² in W·μm⁴/m²
    c2 = 1.4388f4      # h c / k in μm·K

    n = length(CIE_S_LAMBDA)
    lambdas = ntuple(i -> CIE_S_LAMBDA[i], Val(107))
    values = ntuple(Val(107)) do i
        λ_nm = CIE_S_LAMBDA[i]
        λ_um = λ_nm * 1f-3  # nm → μm
        # Planck in W/sr/m²/μm, scaled to match D illuminant magnitude (~100 at 560nm)
        v = c1 / (λ_um^5 * (exp(c2 / (λ_um * T)) - 1f0))
        # Normalize so that 560nm ≈ 100 (matching D65 convention)
        v
    end
    # Normalize to match D illuminant scale (value at 560nm ≈ 1.0 after *0.01)
    v560 = c1 / (0.56f0^5 * (exp(c2 / (0.56f0 * T)) - 1f0))
    scale = 1f0 / v560  # normalize to 1.0 at 560nm (same as D illuminant * 0.01)
    values_scaled = ntuple(i -> values[i] * scale, Val(107))

    return PiecewiseLinearSpectrum{107}(lambdas, values_scaled)
end

"""
    cie_d_illuminant_spectrum(temperature_K) -> PiecewiseLinearSpectrum

Generate a CIE D illuminant spectral power distribution at the given color temperature.
Uses the CIE S0/S1/S2 basis vectors. Falls back to Planck blackbody for T < 4000K.
Matches pbrt-v4's Spectra::D().
"""
function cie_d_illuminant_spectrum(T::Float32)
    # Convert temperature to CCT (pbrt-v4: cct = temperature * 1.4388 / 1.4380)
    cct = T * 1.4388f0 / 1.4380f0

    # CIE D illuminant undefined below 4000K — use Planck blackbody
    if cct < 4000f0
        return planck_blackbody_spectrum(cct)
    end

    # Compute xy chromaticity
    x = if cct <= 7000f0
        -4.607f0 * 1f9 / cct^3 + 2.9678f0 * 1f6 / cct^2 + 0.09911f0 * 1f3 / cct + 0.244063f0
    else
        -2.0064f0 * 1f9 / cct^3 + 1.9018f0 * 1f6 / cct^2 + 0.24748f0 * 1f3 / cct + 0.23704f0
    end
    y = -3f0 * x^2 + 2.870f0 * x - 0.275f0

    # Compute M1, M2 coefficients
    M_denom = 0.0241f0 + 0.2562f0 * x - 0.7341f0 * y
    M1 = (-1.3515f0 - 1.7703f0 * x + 5.9114f0 * y) / M_denom
    M2 = (0.0300f0 - 31.4424f0 * x + 30.0717f0 * y) / M_denom

    # Build spectrum: S(λ) = (S0(λ) + S1(λ)*M1 + S2(λ)*M2) * 0.01
    n = length(CIE_S_LAMBDA)
    lambdas = ntuple(i -> CIE_S_LAMBDA[i], Val(107))
    values = ntuple(i -> (CIE_S0[i] + CIE_S1[i] * M1 + CIE_S2[i] * M2) * 0.01f0, Val(107))

    return PiecewiseLinearSpectrum{107}(lambdas, values)
end

const D65_ILLUMINANT_TABLE = from_interleaved(PiecewiseLinearSpectrum{35}, (
    380.0,49.9755,390.0,54.6482,400.0,82.7549,410.0,91.486,420.0,93.4318,430.0,86.6823,
    440.0,104.865,450.0,117.008,460.0,117.812,470.0,114.861,480.0,115.923,490.0,108.811,
    500.0,109.354,510.0,107.802,520.0,104.79,530.0,107.689,540.0,104.405,550.0,104.046,
    560.0,100.0,570.0,96.3342,580.0,95.788,590.0,88.6856,600.0,90.0062,610.0,89.5991,
    620.0,87.6987,630.0,83.2886,640.0,83.6992,650.0,80.0268,660.0,80.2146,670.0,82.2778,
    680.0,78.2842,690.0,69.7213,700.0,71.6091,710.0,74.349,720.0,61.604,
))
