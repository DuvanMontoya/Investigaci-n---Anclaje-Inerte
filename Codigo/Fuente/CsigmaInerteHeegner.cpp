// C_SIGMA^INERTE — Kernel inerte determinista para discriminantes de Heegner
// Implementación afinada y alineada con el manuscrito (densidades locales, HΔ(R), H⋆Δ(R)).
//
// Compilacion sugerida (GCC/Clang):
//   g++ -O3 -march=native -flto -fopenmp -std=c++17 Codigo/Fuente/CsigmaInerteHeegner.cpp -o Codigo/Binarios/CsigmaHeegner.exe
//   (omitir -fopenmp si no quieres paralelismo)

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Constantes (alta precisión para baseline) ───────────────────────────────
static constexpr long double EULER_GAMMA_LD =
    0.57721566490153286060651209008240243104215933593992L;
static constexpr long double PI_LD =
    3.14159265358979323846264338327950288419716939937510L;
static constexpr long double EXP_MINUS_GAMMA_LD =
    0.56145948356251708990153328994016607745740802432851L; // e^{-γ}

// ── Alias de tipos ──────────────────────────────────────────────────────────
using i32 = std::int32_t;
using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f64 = double;
using ld  = long double;

// ── Globales reutilizables ──────────────────────────────────────────────────
static std::vector<i32> g_spf;      // smallest prime factor
static std::vector<i32> g_primes;   // lista de primos
static std::vector<i32> g_rest;     // rest[x] = x / spf[x]^{e_spf(x)}
static std::vector<u8>  g_par;      // paridad del exponente de spf[x]
static std::vector<u32> g_core;     // sΔ(x): kernel squarefree-inerte (≤ x ≤ LIM)
static std::vector<u8>  g_inert;    // 1 si χΔ(p) = −1
static std::vector<u8>  g_ramif;    // 1 si χΔ(p) =  0

// ── Utilidades IO ──────────────────────────────────────────────────────────
static void eprint(const std::string& s) {
    std::fwrite(s.c_str(), 1, s.size(), stderr);
    std::fflush(stderr);
}

static std::string fmt_ms(long long ms) {
    if (ms < 1000) return std::to_string(ms) + " ms";
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
    return os.str();
}

static std::string hbytes(std::size_t b) {
    static const char* sf[] = {"B","KB","MB","GB"};
    double x = static_cast<double>(b);
    int i = 0;
    while (x >= 1024.0 && i < 3) { x /= 1024.0; ++i; }
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << x << " " << sf[i];
    return os.str();
}

// ── Fase 1: Criba lineal SPF ───────────────────────────────────────────────
static void build_spf(int n) {
    g_spf.assign(n + 1, 0);
    g_primes.clear();
    if (n >= 1) g_spf[1] = 1;

    // Reserva aproximada para evitar reallocs.
    if (n >= 10) {
        const double dn = static_cast<double>(n);
        const double approx = dn / std::log(std::max(3.0, dn)) * 1.15;
        g_primes.reserve(static_cast<std::size_t>(approx));
    }

    for (int i = 2; i <= n; ++i) {
        if (!g_spf[i]) { g_spf[i] = i; g_primes.push_back(i); }
        for (int p : g_primes) {
            const long long v = 1LL * p * i;
            if (v > n) break;
            g_spf[static_cast<int>(v)] = p;
            if (p == g_spf[i]) break;
        }
    }
}

// ── Fase 2: DP rest/par (independiente de Δ) ───────────────────────────────
static void build_rest_par(int n) {
    g_rest.assign(n + 1, 0);
    g_par.assign(n + 1, 0);
    g_rest[1] = 1;
    g_par[1]  = 0;

    const int rpt = std::max(1, n / 20);
    const i32* spf = g_spf.data();
    i32* rest = g_rest.data();
    u8*  par  = g_par.data();

    for (int x = 2; x <= n; ++x) {
        const int p = spf[x];
        const int y = x / p;
        if (y > 1 && spf[y] == p) {
            par[x]  = static_cast<u8>(par[y] ^ 1u);
            rest[x] = rest[y];
        } else {
            par[x]  = 1u;
            rest[x] = y;
        }
        if (x % rpt == 0) {
            std::ostringstream os;
            os << "\r    rest/par: " << std::setw(3) << (100 * x / n) << "%";
            eprint(os.str());
        }
    }
    eprint("\r    rest/par: 100%\n");
}

// ── Símbolo de Kronecker χΔ(p) para p primo ─────────────────────────────────
static long long mod_pow_ll(long long a, long long e, long long m) {
    long long r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = static_cast<long long>((__int128)r * a % m);
        a = static_cast<long long>((__int128)a * a % m);
        e >>= 1;
    }
    return r;
}

static int chi_kronecker(int D, int p) {
    if (p == 2) {
        if (!(D & 1)) return 0; // Δ par → 2 ramificado
        const int r = ((D % 8) + 8) % 8;
        return (r == 1 || r == 7) ? 1 : -1; // Δ≡±1 → split, Δ≡±3 → inerte
    }
    const int a = ((D % p) + p) % p;
    if (!a) return 0;
    return (mod_pow_ll(a, (p - 1) / 2, p) == 1) ? 1 : -1;
}

// ── Clasificación del lugar 2 (Δ mod 8) ────────────────────────────────────
enum class Place2 { RAMIFIED, SPLIT, INERT };

static Place2 classify_2(int D) {
    if (!(D & 1)) return Place2::RAMIFIED;
    const int r = ((D % 8) + 8) % 8;
    return (r == 1 || r == 7) ? Place2::SPLIT : Place2::INERT;
}

// ── Máscaras inerte/ramificado para Δ activo ───────────────────────────────
static void build_prime_masks(int LIM, int D) {
    g_inert.assign(LIM + 1, 0);
    g_ramif.assign(LIM + 1, 0);
    for (int p : g_primes) {
        if (p > LIM) break;
        const int s = chi_kronecker(D, p);
        if (s == -1) g_inert[p] = 1;
        if (s ==  0) g_ramif[p] = 1;
    }
}

// ── Kernel squarefree-inerte sΔ(x) (exacto; sin overflow) ───────────────────
// sΔ(x) = ∏_{p inerte, v_p(x) impar} p.
// Observación clave: sΔ(x) ≤ x para todo x, luego u32 basta si LIM ≤ 2^32−1.
static void build_core(int n) {
    g_core.assign(n + 1, 0u);
    g_core[1] = 1u;

    const int rpt = std::max(1, n / 20);
    const i32* spf   = g_spf.data();
    const i32* rest  = g_rest.data();
    const u8*  par   = g_par.data();
    const u8*  inert = g_inert.data();
    u32* core        = g_core.data();

    for (int x = 2; x <= n; ++x) {
        const int p = spf[x];
        u32 v = core[rest[x]];
        if (inert[p] && par[x]) {
            v = static_cast<u32>(v * static_cast<u32>(p));
        }
        core[x] = v;

        if (x % rpt == 0) {
            std::ostringstream os;
            os << "\r    core sΔ:  " << std::setw(3) << (100 * x / n) << "%";
            eprint(os.str());
        }
    }
    eprint("\r    core sΔ:  100%\n");
}

// ── Densidades locales (Teorema 5.3 y Teorema 6.4) ─────────────────────────
static inline f64 h_p_odd_inert(int p, int m) {
    // h_p(m) = σ_p^(m)/σ_p^(0) con σ_p^(m)=1−2/(p^m(p+1)), σ_p^(0)=(p−1)/(p+1)
    // => h_p(m) = [p^m(p+1)−2]/[p^m(p−1)]
    f64 pm = 1.0;
    for (int i = 0; i < m; ++i) pm *= static_cast<f64>(p);
    return (pm * (p + 1) - 2.0) / (pm * (p - 1));
}

static inline f64 h_2_inert(int m) {
    // σ_2^(m)=1−1/(3·2^m), σ_2^(0)=2/3 => h_2(m)=(3·2^m−1)/2^{m+1}
    const f64 pm = static_cast<f64>(1ULL << m);
    return (3.0 * pm - 1.0) / (2.0 * pm);
}

struct HPair {
    f64 H;       // HΔ(R)
    f64 Hstar;   // H⋆Δ(R) = HΔ(R) * ∏_{p|Δ, p impar} p^{-v_p(R)}
    f64 ram_prod; // ∏_{p|Δ, p impar} p^{v_p(R)} (útil porque C⋆ = C * ram_prod)
};

static inline HPair compute_H_pair(int R, Place2 p2type, bool want_Hstar) {
    f64 H = 1.0;
    f64 ram_prod = 1.0;

    int n = R;

    // Manejo de 2: solo contribuye a HΔ si 2 es inerte y v2(R)>0.
    if (p2type == Place2::INERT) {
        int m2 = 0;
#if defined(__GNUG__) || defined(__clang__)
        m2 = __builtin_ctz(static_cast<unsigned>(n));
        n >>= m2;
#else
        while ((n & 1) == 0) { ++m2; n >>= 1; }
#endif
        if (m2 > 0) H *= h_2_inert(m2);
    } else {
        while ((n & 1) == 0) n >>= 1;
    }

    // Parte impar: factorización por SPF.
    const i32* spf = g_spf.data();
    const u8* inert = g_inert.data();
    const u8* ramif = g_ramif.data();

    while (n > 1) {
        const int p = spf[n];
        int mv = 0;
        do { n /= p; ++mv; } while (n > 1 && spf[n] == p);

        if (inert[p]) H *= h_p_odd_inert(p, mv);

        // Refinamiento ramificado: solo p impar, y solo si p|Δ (ramificado).
        if (want_Hstar && ramif[p]) {
            f64 pv = 1.0;
            for (int i = 0; i < mv; ++i) pv *= static_cast<f64>(p);
            ram_prod *= pv;
        }
    }

    const f64 Hstar = want_Hstar ? (H / ram_prod) : H;
    return {H, Hstar, ram_prod};
}

// ── Conteo determinista NΔ(R) usando kernel ─────────────────────────────────
// NΔ(R) = #{1 ≤ c ≤ R−1 : sΔ(R−c) == sΔ(R+c)}
static inline int compute_N(int R, const u32* __restrict__ core) {
    int cnt = 0;
    const u32* a = core + (R - 1);
    const u32* b = core + (R + 1);
    const int iters = R - 1;
    int i = 0;

    for (; i + 7 < iters; i += 8) {
        cnt += (a[ 0] == b[0]);
        cnt += (a[-1] == b[1]);
        cnt += (a[-2] == b[2]);
        cnt += (a[-3] == b[3]);
        cnt += (a[-4] == b[4]);
        cnt += (a[-5] == b[5]);
        cnt += (a[-6] == b[6]);
        cnt += (a[-7] == b[7]);
        a -= 8; b += 8;
    }
    for (; i < iters; ++i, --a, ++b) cnt += (*a == *b);
    return cnt;
}

// ── Estadística básica ─────────────────────────────────────────────────────
static f64 vec_mean(const std::vector<f64>& v) {
    if (v.empty()) return std::numeric_limits<f64>::quiet_NaN();
    ld acc = 0;
    for (f64 x : v) acc += static_cast<ld>(x);
    return static_cast<f64>(acc / static_cast<ld>(v.size()));
}

static f64 vec_pstdev(const std::vector<f64>& v, f64 mu) {
    if (v.empty()) return std::numeric_limits<f64>::quiet_NaN();
    ld acc = 0;
    for (f64 x : v) {
        const ld d = static_cast<ld>(x) - static_cast<ld>(mu);
        acc += d * d;
    }
    return static_cast<f64>(std::sqrt(acc / static_cast<ld>(v.size())));
}

static f64 vec_median(std::vector<f64> v) {
    if (v.empty()) return std::numeric_limits<f64>::quiet_NaN();
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    f64 med = v[m];
    if (!(v.size() & 1)) {
        std::nth_element(v.begin(), v.begin() + (m - 1), v.end());
        med = 0.5 * (med + v[m - 1]);
    }
    return med;
}

// ── Ajuste de cola (Sección 9.3): WLS binned en cuantiles ───────────────────
// Modelo: y = C∞ + a·x, con x = 1/log R, usando R ≥ R_tail.
// Binning opcional en 'bins' cuantiles; pesos w = tamaño del bin.
struct TailFit { f64 Cinf, a, se_Cinf, se_a; std::size_t n; };
static const TailFit NaN_TF{std::numeric_limits<f64>::quiet_NaN(),
                            std::numeric_limits<f64>::quiet_NaN(),
                            std::numeric_limits<f64>::quiet_NaN(),
                            std::numeric_limits<f64>::quiet_NaN(),
                            0};

static TailFit tail_fit_wls_binned(const std::vector<f64>& Rv,
                                  const std::vector<f64>& Cv,
                                  f64 R_tail,
                                  int bins) {
    std::vector<std::size_t> idx;
    idx.reserve(Rv.size());
    for (std::size_t i = 0; i < Rv.size(); ++i) {
        if (Rv[i] >= R_tail && !std::isnan(Cv[i])) idx.push_back(i);
    }
    if (idx.size() < 4) return NaN_TF;

    struct Pt { ld x, y, w; };
    std::vector<Pt> pts;

    if (bins <= 0 || idx.size() <= static_cast<std::size_t>(bins)) {
        pts.reserve(idx.size());
        for (std::size_t j : idx) {
            const ld x = 1.0L / std::log(static_cast<ld>(Rv[j]));
            pts.push_back({x, static_cast<ld>(Cv[j]), 1.0L});
        }
    } else {
        const int B = std::max(1, bins);
        pts.reserve(static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) {
            const std::size_t lo = (static_cast<std::size_t>(b) * idx.size()) / B;
            const std::size_t hi = (static_cast<std::size_t>(b + 1) * idx.size()) / B;
            if (hi <= lo) continue;

            ld sx = 0, sy = 0;
            for (std::size_t t = lo; t < hi; ++t) {
                const std::size_t j = idx[t];
                sx += 1.0L / std::log(static_cast<ld>(Rv[j]));
                sy += static_cast<ld>(Cv[j]);
            }
            const ld w = static_cast<ld>(hi - lo);
            pts.push_back({sx / w, sy / w, w});
        }
    }

    const std::size_t n = pts.size();
    if (n < 4) return NaN_TF;

    ld Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (const auto& p : pts) {
        Sw  += p.w;
        Sx  += p.w * p.x;
        Sy  += p.w * p.y;
        Sxx += p.w * p.x * p.x;
        Sxy += p.w * p.x * p.y;
    }

    const ld det = Sw * Sxx - Sx * Sx;
    if (std::abs(det) < 1e-30L) return NaN_TF;

    const ld Cinf = (Sy * Sxx - Sx * Sxy) / det;
    const ld a    = (Sw * Sxy - Sx * Sy) / det;

    ld SSE = 0;
    for (const auto& p : pts) {
        const ld res = p.y - (Cinf + a * p.x);
        SSE += p.w * res * res;
    }

    const ld dof = static_cast<ld>(n) - 2.0L;
    if (dof <= 0) return NaN_TF;
    const ld mse = SSE / dof;

    const f64 se_Cinf = static_cast<f64>(std::sqrt(mse * Sxx / det));
    const f64 se_a    = static_cast<f64>(std::sqrt(mse * Sw  / det));

    return {static_cast<f64>(Cinf), static_cast<f64>(a), se_Cinf, se_a, n};
}

// ── Baseline canónico en Heegner (Teorema 7.3) ─────────────────────────────
static int heegner_w(int D) {
    return (D == -3) ? 6 : (D == -4) ? 4 : 2;
}

static f64 L1_heegner(int D) {
    const int w = heegner_w(D);
    return static_cast<f64>(2.0L * PI_LD / (static_cast<ld>(w) * std::sqrt(static_cast<ld>(std::abs(D)))));
}

static f64 Cbase_heegner(int D) {
    return static_cast<f64>(2.0L * EXP_MINUS_GAMMA_LD * static_cast<ld>(L1_heegner(D)));
}

static const std::array<int, 9> HEEGNER_D = {-3,-4,-7,-8,-11,-19,-43,-67,-163};
static bool is_heegner(int D) {
    return std::find(HEEGNER_D.begin(), HEEGNER_D.end(), D) != HEEGNER_D.end();
}

// ── Argumentos CLI ─────────────────────────────────────────────────────────
struct Args {
    int Rmax = 1000001;
    int Rmin = 101;
    int step = 1;
    f64 R_tail = 1e6;

    bool all = false;
    std::vector<int> Ds = {-4};

    std::string outdir  = "Datos/Canonicos";
    std::string prefix  = "csigma_inerte";
    std::string summary = "ResumenHeegner.csv";

    bool compute_Hstar = true;   // exporta también C⋆σ,Δ
    bool do_tail_fit   = true;
    int  tail_bins     = 200;    // Sección 9.3: binning en cuantiles (tabla usa 200 bins)
    bool progress      = true;
};

static void usage(const char* p) {
    std::cerr
      << "Uso: " << p << " [opciones]\n\n"
      << "  --D  D           discriminante fundamental negativo (p.ej. -4)\n"
      << "  --Ds d1,d2       lista separada por comas de discriminantes\n"
      << "  --all            los 9 discriminantes de Heegner\n"
      << "  --Rmax N         radio impar máximo [default 1000001]\n"
      << "  --Rmin M         radio mínimo exportado [default 101]\n"
      << "  --step S         paso en el índice impar [default 1]\n"
      << "  --Rtail T        umbral R para ajuste de cola (9.3) [default 1e6]\n"
      << "  --bins B         #bins cuantiles para WLS (9.3) [default 200]\n"
      << "  --no-bins        desactiva binning (WLS punto-a-punto)\n"
      << "  --outdir DIR     carpeta de salida [default Datos/Canonicos]\n"
      << "  --prefix P       prefijo de CSV [default csigma_inerte]\n"
      << "  --summary F      nombre del CSV resumen [default ResumenHeegner.csv]\n"
      << "  --no-Hstar       omitir C⋆σ,Δ (refinamiento ramificado)\n"
      << "  --no-tail        omitir ajuste de cola\n"
      << "  --no-progress    desactivar barra de progreso\n"
      << "  -h / --help      este mensaje\n";
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); }
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));

    std::vector<int> uniq;
    for (int d : out) {
        if (std::find(uniq.begin(), uniq.end(), d) == uniq.end()) uniq.push_back(d);
    }
    return uniq;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto need = [&](const char* nm) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para " << nm << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if      (s == "--Rmax")     a.Rmax       = std::stoi(need("--Rmax"));
        else if (s == "--Rmin")     a.Rmin       = std::stoi(need("--Rmin"));
        else if (s == "--step")     a.step       = std::stoi(need("--step"));
        else if (s == "--Rtail")    a.R_tail     = std::stod(need("--Rtail"));
        else if (s == "--bins")     a.tail_bins  = std::stoi(need("--bins"));
        else if (s == "--no-bins")  a.tail_bins  = 0;
        else if (s == "--D")        a.Ds         = {std::stoi(need("--D"))};
        else if (s == "--Ds")       a.Ds         = parse_int_list(need("--Ds"));
        else if (s == "--all")      a.all        = true;
        else if (s == "--outdir")   a.outdir     = need("--outdir");
        else if (s == "--prefix")   a.prefix     = need("--prefix");
        else if (s == "--summary")  a.summary    = need("--summary");
        else if (s == "--no-Hstar") a.compute_Hstar = false;
        else if (s == "--no-tail")  a.do_tail_fit   = false;
        else if (s == "--no-progress") a.progress = false;
        else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
        else {
            std::cerr << "Argumento desconocido: " << s << "\n";
            usage(argv[0]);
            std::exit(2);
        }
    }

    if (a.Rmax < 1) a.Rmax = 1;
    if (!(a.Rmax & 1)) --a.Rmax; // forzar impar
    if (a.Rmin < 1) a.Rmin = 1;
    if (a.step < 1) a.step = 1;

    if (a.all) a.Ds.assign(HEEGNER_D.begin(), HEEGNER_D.end());
    if (a.Ds.empty()) a.Ds = {-4};

    if (a.tail_bins < 0) a.tail_bins = 0;

    return a;
}

// ── MAIN ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    using clk = std::chrono::high_resolution_clock;
    using ms  = std::chrono::milliseconds;

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args args = parse_args(argc, argv);

    for (int D : args.Ds) {
        if (!is_heegner(D)) {
            eprint("[ADVERTENCIA] D=" + std::to_string(D) + " no es Heegner (h≠1).\n"
                   "  Las densidades locales son correctas, pero el paso global (ideales vs elementos)\n"
                   "  puede fallar fuera de h=1.\n");
        }
    }

    const int  R_MAX  = args.Rmax;
    const int  R_MIN  = args.Rmin;
    const int  STEP   = args.step;
    const int  LIM    = 2 * R_MAX + 2;
    const f64  R_TAIL = args.R_tail;

    // Grid de radios impares
    const int total_odd_half = (R_MAX + 1) / 2;
    const int total_R = (total_odd_half + STEP - 1) / STEP;
    std::vector<int> R_vals(total_R);
    for (int k = 0; k < total_R; ++k) R_vals[k] = 2 * (k * STEP) + 1;

    // Precomputo log(R) para el barrido
    std::vector<f64> logR_vals(total_R);
    for (int k = 0; k < total_R; ++k) logR_vals[k] = std::log(static_cast<f64>(R_vals[k]));

    // Estimación memoria
    const std::size_t mem_spf   = static_cast<std::size_t>(LIM + 1) * sizeof(i32);
    const std::size_t mem_rest  = static_cast<std::size_t>(LIM + 1) * sizeof(i32);
    const std::size_t mem_par   = static_cast<std::size_t>(LIM + 1) * sizeof(u8);
    const std::size_t mem_core  = static_cast<std::size_t>(LIM + 1) * sizeof(u32);
    const std::size_t mem_masks = static_cast<std::size_t>(LIM + 1) * sizeof(u8) * 2;
    const std::size_t mem_vals  = static_cast<std::size_t>(total_R) * (sizeof(int) + sizeof(f64) * 6);
    const std::size_t mem_total = mem_spf + mem_rest + mem_par + mem_core + mem_masks + mem_vals;

    // Cabecera
    {
        std::ostringstream os;
        os << "\n╔═══════════════════════════════════════════════════════════════╗\n"
           << "║  C_σ^INERTE — Heegner h=1 (alineado al manuscrito)           ║\n"
           << "╚═══════════════════════════════════════════════════════════════╝\n"
           << "  R_MAX      = " << R_MAX << "\n"
           << "  LIM        = " << LIM   << "\n"
           << "  STEP       = " << STEP  << "  → " << total_R << " radios\n"
           << "  R_MIN exp. = " << R_MIN << "\n"
           << "  R_tail fit = " << static_cast<long long>(R_TAIL) << "\n"
           << "  bins (9.3) = " << args.tail_bins << "\n"
           << "  D(s)       = ";
        for (std::size_t i = 0; i < args.Ds.size(); ++i)
            os << args.Ds[i] << (i + 1 == args.Ds.size() ? "" : ", ");
        os << "\n";
#ifdef _OPENMP
        os << "  OpenMP     = ON (" << omp_get_max_threads() << " threads)\n";
#else
        os << "  OpenMP     = OFF (compilar con -fopenmp para paralelismo)\n";
#endif
        os << "  Memoria    ≈ " << hbytes(mem_total) << "\n"
           << "  Kernel sΔ  = uint32_t (sΔ(x) ≤ x)\n"
           << "  C⋆σ,Δ (ramificado) = " << (args.compute_Hstar ? "SÍ" : "NO") << "\n"
           << "  Ajuste cola (9.3)  = " << (args.do_tail_fit ? "SÍ" : "NO") << "\n"
           << "───────────────────────────────────────────────────────────────\n";
        eprint(os.str());
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(args.outdir, ec);
    if (ec) {
        std::cerr << "No puedo crear outdir='" << args.outdir << "': " << ec.message() << "\n";
        return 3;
    }

    const auto T_total = clk::now();

    try {
        // ── Fases globales ────────────────────────────────────────────────
        eprint(">>> [Fase 1] Criba lineal SPF hasta LIM=" + std::to_string(LIM) + "... ");
        {
            const auto t = clk::now();
            build_spf(LIM);
            eprint("OK [" + fmt_ms(std::chrono::duration_cast<ms>(clk::now() - t).count()) + "]\n");
        }

        eprint(">>> [Fase 2] DP rest/par (independiente de Δ)...\n");
        {
            const auto t = clk::now();
            build_rest_par(LIM);
            eprint("    OK [" + fmt_ms(std::chrono::duration_cast<ms>(clk::now() - t).count()) + "]\n");
        }

        // ── Summary ───────────────────────────────────────────────────────
        const std::string sum_path = (fs::path(args.outdir) / args.summary).string();
        std::ofstream fsum(sum_path);
        if (!fsum) {
            std::cerr << "No puedo crear summary: " << sum_path << "\n";
            return 4;
        }

        fsum << std::fixed << std::setprecision(12);
        fsum << "D,p2_type,Rmax,step,Rmin,Rtail,bins_tail,pts_export,"
             << "C_median,C_mean,C_sd,"
             << "C_inf,a_slope,se_Cinf,se_a,n_tail,"
             << "C_base,L1_chi,S_Delta,"
             << "C_inf_star,a_star,se_Cinf_star,se_a_star,n_tail_star,"
             << "csv_file\n";

        // ── Loop por discriminante ────────────────────────────────────────
        for (std::size_t idxD = 0; idxD < args.Ds.size(); ++idxD) {
            const int D = args.Ds[idxD];
            const Place2 p2type = classify_2(D);
            const bool heegner = is_heegner(D);

            const std::string p2str =
                (p2type == Place2::INERT) ? "inert" :
                (p2type == Place2::SPLIT) ? "split" : "ramif";

            {
                std::ostringstream os;
                os << "\n╔═══════════════════════════════════════════════════════════════╗\n"
                   << "║  D = " << std::setw(5) << D
                   << "   lugar-2: " << std::left << std::setw(6) << p2str;
                os << (heegner ? "  [Heegner h=1]" : "  [NO Heegner]  ");
                os << "  (" << (idxD + 1) << "/" << args.Ds.size() << ")\n"
                   << "╚═══════════════════════════════════════════════════════════════╝\n";
                eprint(os.str());
            }

            // D1: máscaras
            eprint("  [D1] Máscaras Kronecker χΔ... ");
            {
                const auto t = clk::now();
                build_prime_masks(LIM, D);
                eprint("OK [" + fmt_ms(std::chrono::duration_cast<ms>(clk::now() - t).count()) + "]\n");
            }

            // Diagnóstico: inertes pequeños
            {
                std::vector<int> small_inerts;
                for (int p : g_primes) {
                    if (p > 100) break;
                    if (g_inert[p]) small_inerts.push_back(p);
                }
                std::ostringstream os;
                os << "       Inertes p≤100: {";
                for (std::size_t i = 0; i < small_inerts.size(); ++i)
                    os << small_inerts[i] << (i + 1 < small_inerts.size() ? "," : "");
                os << "}";
                f64 prod = 1.0;
                for (int p : small_inerts) prod *= static_cast<f64>(p - 1) / static_cast<f64>(p + 1);
                os << "  ∏(p−1)/(p+1) = " << std::fixed << std::setprecision(6) << prod << "\n";
                eprint(os.str());
            }

            // D2: kernel
            eprint("  [D2] Kernel sΔ(x) en uint32_t...\n");
            {
                const auto t = clk::now();
                build_core(LIM);
                eprint("       OK [" + fmt_ms(std::chrono::duration_cast<ms>(clk::now() - t).count()) + "]\n");
            }

            // D3: barrido en R
            eprint("  [D3] Barrido en R" + std::string(args.progress ? " (progreso)" : "") + "...\n");

            std::vector<int> N_vals(total_R, 0);
            std::vector<f64> H_vals(total_R, 0.0);
            std::vector<f64> Hs_vals(total_R, 0.0);
            std::vector<f64> C_vals(total_R, std::numeric_limits<f64>::quiet_NaN());
            std::vector<f64> Cs_vals(total_R, std::numeric_limits<f64>::quiet_NaN());

            const u32* core = g_core.data();

            std::atomic<int> done{0};
            std::atomic<bool> stop_rep{false};
            std::thread reporter;
            if (args.progress) {
                reporter = std::thread([&] {
                    int last = -1;
                    while (!stop_rep.load(std::memory_order_relaxed)) {
                        const int cur = done.load(std::memory_order_relaxed);
                        const int pct = total_R ? static_cast<int>(100LL * cur / total_R) : 100;
                        if (pct != last) {
                            std::ostringstream os;
                            os << "\r       progreso: " << std::setw(3) << pct << "%";
                            eprint(os.str());
                            last = pct;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                });
            }

            const auto t_sweep = clk::now();

#ifdef _OPENMP
            #pragma omp parallel for schedule(guided)
#endif
            for (int k = 0; k < total_R; ++k) {
                const int R = R_vals[k];
                const int N = compute_N(R, core);

                const HPair hp = compute_H_pair(R, p2type, args.compute_Hstar);
                N_vals[k] = N;
                H_vals[k] = hp.H;
                Hs_vals[k] = hp.Hstar;

                if (R > 1) {
                    const f64 logR = logR_vals[k];
                    if (hp.H > 0.0) {
                        const f64 C = static_cast<f64>(N) * logR / (hp.H * static_cast<f64>(R));
                        C_vals[k] = C;

                        if (args.compute_Hstar && hp.Hstar > 0.0) {
                            // Importante: por definición H⋆ = H * ∏ p^{-vp(R)}, luego
                            // C⋆ = N logR/(H⋆R) = C * ∏ p^{vp(R)} = C * ram_prod.
                            Cs_vals[k] = C * hp.ram_prod;
                        }
                    }
                }

                ++done;
            }

            stop_rep.store(true, std::memory_order_relaxed);
            if (args.progress) {
                reporter.join();
                eprint("\r       progreso: 100%\n");
            }

            {
                const auto t_ms = std::chrono::duration_cast<ms>(clk::now() - t_sweep).count();
                std::ostringstream os;
                os << "       OK [" << fmt_ms(t_ms) << "]  "
                   << std::fixed << std::setprecision(1)
                   << (t_ms > 0 ? 1e3 * total_R / t_ms : 0.0) << " R/s\n";
                eprint(os.str());
            }

            // Estadísticas en R≥R_MIN
            std::vector<f64> C_used;
            C_used.reserve(total_R);
            for (int k = 0; k < total_R; ++k) {
                if (R_vals[k] >= R_MIN && !std::isnan(C_vals[k])) C_used.push_back(C_vals[k]);
            }

            const f64 C_med = vec_median(C_used);
            const f64 C_avg = vec_mean(C_used);
            const f64 C_sd  = vec_pstdev(C_used, C_avg);

            // Tail-fit usa series completas
            std::vector<f64> Rv_all(total_R), Cv_all(total_R), Csv_all(total_R);
            for (int k = 0; k < total_R; ++k) {
                Rv_all[k] = static_cast<f64>(R_vals[k]);
                Cv_all[k] = C_vals[k];
                Csv_all[k] = Cs_vals[k];
            }

            TailFit tf = NaN_TF;
            TailFit tfs = NaN_TF;
            if (args.do_tail_fit) {
                tf = tail_fit_wls_binned(Rv_all, Cv_all, R_TAIL, args.tail_bins);
                if (args.compute_Hstar) tfs = tail_fit_wls_binned(Rv_all, Csv_all, R_TAIL, args.tail_bins);
            }

            // Baseline Heegner
            const f64 Cbase   = heegner ? Cbase_heegner(D) : std::numeric_limits<f64>::quiet_NaN();
            const f64 L1      = heegner ? L1_heegner(D)    : std::numeric_limits<f64>::quiet_NaN();
            const f64 S_Delta = (heegner && Cbase > 0 && !std::isnan(tf.Cinf)) ? (tf.Cinf / Cbase)
                              : std::numeric_limits<f64>::quiet_NaN();

            // Resumen consola
            {
                std::ostringstream os;
                os << std::fixed << std::setprecision(9);
                os << "  ┌─ RESUMEN  D=" << D << "  (R≥" << R_MIN << ", " << C_used.size() << " pts) ───────\n"
                   << "  │  mediana C(R)    = " << C_med << "\n"
                   << "  │  media   C(R)    = " << C_avg << "\n"
                   << "  │  σ empírica      = " << C_sd  << "\n";
                if (!std::isnan(tf.Cinf)) {
                    os << "  │  C_∞ (WLS, 9.3) = " << tf.Cinf << " ± " << tf.se_Cinf
                       << "  (n=" << tf.n << ")\n"
                       << "  │  a (1/log)      = " << tf.a    << " ± " << tf.se_a << "\n";
                }
                if (heegner) {
                    os << "  │  C_base(Δ)      = " << Cbase  << "   [4πe^{-γ}/(w√|Δ|)]\n"
                       << "  │  L(1, χ_Δ)      = " << L1     << "\n";
                    if (!std::isnan(S_Delta)) os << "  │  S_Δ = C_∞/Cbase = " << S_Delta << "\n";
                }
                if (!std::isnan(tfs.Cinf)) {
                    os << "  │  C_∞⋆ (ramif)   = " << tfs.Cinf << " ± " << tfs.se_Cinf
                       << "  (n=" << tfs.n << ")\n";
                }
                os << "  │  lugar-2         = " << p2str;
                if (p2type == Place2::INERT) os << "  (incluye h₂(m) si v₂(R)>0)";
                os << "\n  └──────────────────────────────────────────────\n";
                eprint(os.str());
            }

            // Export CSV por D
            const std::string csv_name = args.prefix
                + "_D" + std::to_string(D)
                + "_Rmax" + std::to_string(R_MAX)
                + "_step" + std::to_string(STEP)
                + ".csv";
            const std::string csv_path = (fs::path(args.outdir) / csv_name).string();
            {
                std::ofstream fout(csv_path);
                if (!fout) { std::cerr << "No puedo abrir: " << csv_path << "\n"; return 5; }
                fout << std::fixed << std::setprecision(12);

                fout << "D,R,N_R,H_R";
                if (args.compute_Hstar) fout << ",Hstar_R";
                fout << ",C_sigma";
                if (args.compute_Hstar) fout << ",Cstar_sigma";
                fout << "\n";

                for (int k = 0; k < total_R; ++k) {
                    const int R = R_vals[k];
                    if (R < R_MIN || std::isnan(C_vals[k])) continue;
                    fout << D << "," << R << "," << N_vals[k] << "," << H_vals[k];
                    if (args.compute_Hstar) fout << "," << Hs_vals[k];
                    fout << "," << C_vals[k];
                    if (args.compute_Hstar) fout << "," << Cs_vals[k];
                    fout << "\n";
                }
            }
            eprint("  [OK] CSV escrito: " + csv_path + "\n");

            // Línea summary
            fsum
              << D << "," << p2str << ","
              << R_MAX << "," << STEP << "," << R_MIN << ","
              << static_cast<long long>(R_TAIL) << "," << args.tail_bins << ","
              << C_used.size() << ","
              << C_med << "," << C_avg << "," << C_sd << ","
              << tf.Cinf << "," << tf.a << "," << tf.se_Cinf << "," << tf.se_a << "," << tf.n << ","
              << Cbase << "," << L1 << "," << S_Delta << ","
              << tfs.Cinf << "," << tfs.a << "," << tfs.se_Cinf << "," << tfs.se_a << "," << tfs.n << ","
              << csv_name << "\n";
            fsum.flush();
        }

        // Fin
        {
            const auto t_ms = std::chrono::duration_cast<ms>(clk::now() - T_total).count();
            std::ostringstream os;
            os << "\n╔═══════════════════════════════════════════════════════════════╗\n"
               << "║  FIN  — Tiempo total: " << fmt_ms(t_ms)
               << std::string(std::max(0, 44 - static_cast<int>(fmt_ms(t_ms).size())), ' ') << "║\n"
               << "║  Summary: " << sum_path
               << std::string(std::max(0, 52 - static_cast<int>(sum_path.size())), ' ') << "║\n"
               << "╚═══════════════════════════════════════════════════════════════╝\n";
            eprint(os.str());
        }

        return 0;

    } catch (const std::bad_alloc& e) {
        std::cerr << "\n[ERROR] Memoria insuficiente: " << e.what() << "\n"
                  << "Sugerencias: --step mayor, --Rmax menor, compilar 64-bit.\n";
        return 10;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Excepción: " << e.what() << "\n";
        return 11;
    }
}
