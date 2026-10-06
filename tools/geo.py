"""WGS84 latitude/longitude to UTM and back, with no dependencies. Good to a millimetre."""
import math

A = 6378137.0
F = 1 / 298.257223563
K0 = 0.9996


def _series():
    n = F / (2 - F)
    a = A / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    alpha = [n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5, 61 * n**3 / 240]
    beta = [n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96, n**2 / 48 + n**3 / 15, 17 * n**3 / 480]
    delta = [2 * n - 2 * n**2 / 3 - 2 * n**3, 7 * n**2 / 3 - 8 * n**3 / 5, 56 * n**3 / 15]
    return n, a, alpha, beta, delta


def to_utm(lat, lon, zone):
    """(easting, northing) in metres for a northern-hemisphere UTM zone."""
    n, a, alpha, _, _ = _series()
    lon0 = math.radians(zone * 6 - 183)
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    s = 2 * math.sqrt(n) / (1 + n)
    t = math.sinh(math.atanh(math.sin(phi)) - s * math.atanh(s * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    e = 500000 + K0 * a * (eta + sum(alpha[j] * math.cos(2 * (j + 1) * xi) * math.sinh(2 * (j + 1) * eta) for j in range(3)))
    nn = K0 * a * (xi + sum(alpha[j] * math.sin(2 * (j + 1) * xi) * math.cosh(2 * (j + 1) * eta) for j in range(3)))
    return e, nn


def from_utm(easting, northing, zone):
    """(lat, lon) in degrees for a northern-hemisphere UTM zone."""
    n, a, _, beta, delta = _series()
    xi = northing / (K0 * a)
    eta = (easting - 500000) / (K0 * a)
    xi1 = xi - sum(beta[j] * math.sin(2 * (j + 1) * xi) * math.cosh(2 * (j + 1) * eta) for j in range(3))
    eta1 = eta - sum(beta[j] * math.cos(2 * (j + 1) * xi) * math.sinh(2 * (j + 1) * eta) for j in range(3))
    chi = math.asin(math.sin(xi1) / math.cosh(eta1))
    lat = chi + sum(delta[j] * math.sin(2 * (j + 1) * chi) for j in range(3))
    lon = zone * 6 - 183 + math.degrees(math.atan2(math.sinh(eta1), math.cos(xi1)))
    return math.degrees(lat), lon


if __name__ == "__main__":
    e, nn = to_utm(43.2835697, -89.1026535, 16)
    print(f"{e:.2f} {nn:.2f}", from_utm(e, nn, 16))
