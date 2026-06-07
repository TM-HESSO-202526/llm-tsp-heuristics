#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

struct Point { double x=0, y=0; };
struct Instance { string name, split, edge_type; double opt=0; vector<Point> p; };

struct Deadline {
    chrono::steady_clock::time_point t0;
    double timeout_s;
    explicit Deadline(double s): t0(chrono::steady_clock::now()), timeout_s(s) {}
    bool expired() const {
        return timeout_s > 0 && chrono::duration<double>(chrono::steady_clock::now() - t0).count() > timeout_s;
    }
    void check() const {
        if (expired()) throw runtime_error("timeout");
    }
};

static string trim(string s) {
    auto notsp = [](int c){ return !std::isspace(c); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), notsp));
    s.erase(find_if(s.rbegin(), s.rend(), notsp).base(), s.end());
    return s;
}

static vector<string> split_csv(const string& s) {
    vector<string> r; string cur; stringstream ss(s);
    while (getline(ss, cur, ',')) { cur = trim(cur); if (!cur.empty()) r.push_back(cur); }
    return r;
}

static bool contains(const vector<string>& v, const string& x) {
    return find(v.begin(), v.end(), x) != v.end();
}

static string csv_escape(const string& s) {
    if (s.find_first_of(",\n\r\"") == string::npos) return s;
    string o = "\"";
    for (char c : s) { if (c == '\"') o += "\"\""; else o += c; }
    o += '\"';
    return o;
}

static long long edge_cost(const Instance& inst, int i, int j) {
    const auto &a = inst.p.at(i), &b = inst.p.at(j);
    double dx = a.x - b.x, dy = a.y - b.y;
    double d = sqrt(dx*dx + dy*dy);
    string t = inst.edge_type;
    for (char &c : t) c = (char)toupper(c);
    if (t.find("CEIL") != string::npos) return (long long)ceil(d);
    return (long long)floor(d + 0.5); // TSPLIB EUC_2D convention.
}

static double euclidean_norm(const Instance& inst, int i, int j) {
    const auto &a = inst.p.at(i), &b = inst.p.at(j);
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

static double tour_cost(const Instance& inst, const vector<int>& tour) {
    long double s = 0;
    int n = (int)tour.size();
    for (int i = 0; i < n; i++) s += edge_cost(inst, tour[i], tour[(i+1)%n]);
    return (double)s;
}

static void validate_tour(const vector<int>& tour, int n) {
    if ((int)tour.size() != n) throw runtime_error("invalid tour length");
    vector<char> seen(n, 0);
    for (int x : tour) {
        if (x < 0 || x >= n) throw runtime_error("tour city out of range");
        if (seen[x]) throw runtime_error("duplicate city in tour");
        seen[x] = 1;
    }
}

static Instance read_tsp(const fs::path& pth, const string& name, const string& split, double opt) {
    ifstream f(pth);
    if (!f) throw runtime_error("cannot open tsp: " + pth.string());
    string line, edge = "EUC_2D";
    int dim = -1;
    bool coords = false;
    vector<Point> pts;
    while (getline(f, line)) {
        line = trim(line);
        if (line.empty()) continue;
        string u = line;
        for (char &c : u) c = (char)toupper(c);
        if (u.rfind("EDGE_WEIGHT_TYPE", 0) == 0) {
            auto pos = line.find(':');
            edge = trim(pos == string::npos ? line.substr(16) : line.substr(pos+1));
        }
        if (u.rfind("DIMENSION", 0) == 0) {
            auto pos = line.find(':');
            dim = stoi(trim(pos == string::npos ? line.substr(9) : line.substr(pos+1)));
            pts.reserve(dim);
        }
        if (u == "NODE_COORD_SECTION") { coords = true; continue; }
        if (u == "EOF") break;
        if (coords) {
            stringstream ss(line);
            int id; double x, y;
            if (ss >> id >> x >> y) pts.push_back({x, y});
        }
    }
    if (dim > 0 && (int)pts.size() != dim) cerr << "WARNING: " << name << " dimension " << dim << " but read " << pts.size() << " coords\n";
    if (pts.empty()) throw runtime_error("no coordinates read: " + pth.string());
    return {name, split, edge, opt, move(pts)};
}

struct OptRow { string instance, split; double opt; };

static vector<OptRow> read_opt_csv(const fs::path& p) {
    ifstream f(p);
    if (!f) throw runtime_error("cannot open opt csv");
    string line;
    getline(f, line);
    vector<OptRow> rows;
    while (getline(f, line)) {
        if (trim(line).empty()) continue;
        vector<string> c; string cur; stringstream ss(line);
        while (getline(ss, cur, ',')) c.push_back(trim(cur));
        if (c.size() >= 3) rows.push_back({c[0], c[1], stod(c[2])});
    }
    return rows;
}

// This is not NumPy PCG64 byte-for-byte. It only plays the same role as rng.integers().
// The heuristic structure below is otherwise kept as a direct C++ translation.
struct RNG {
    mt19937_64 gen;
    explicit RNG(uint64_t seed): gen(seed) {}
    int randint(int n) { uniform_int_distribution<int> d(0, n-1); return d(gen); }
    int randint_range(int lo, int hi_exclusive) { uniform_int_distribution<int> d(lo, hi_exclusive-1); return d(gen); }
};

// -----------------------------------------------------------------------------
// Baseline helpers. These are baseline definitions, not LLM heuristic translations.
// -----------------------------------------------------------------------------
static vector<int> x_axis_sweep(const Instance& inst) {
    int n = (int)inst.p.size();
    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&](int a, int b){
        if (inst.p[a].x == inst.p[b].x) return inst.p[a].y < inst.p[b].y;
        return inst.p[a].x < inst.p[b].x;
    });
    return idx;
}

static vector<int> angular_sweep(const Instance& inst) {
    int n = (int)inst.p.size();
    double cx = 0, cy = 0;
    for (auto &p : inst.p) { cx += p.x; cy += p.y; }
    cx /= n; cy /= n;
    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&](int a, int b){
        double aa = atan2(inst.p[a].y - cy, inst.p[a].x - cx);
        double ab = atan2(inst.p[b].y - cy, inst.p[b].x - cx);
        if (aa == ab) {
            double ra = hypot(inst.p[a].x - cx, inst.p[a].y - cy);
            double rb = hypot(inst.p[b].x - cx, inst.p[b].y - cy);
            return ra < rb;
        }
        return aa < ab;
    });
    return idx;
}

static uint64_t norm_key(double v, double lo, double hi, int bits=21) {
    if (hi <= lo) return 0;
    double z = (v-lo)/(hi-lo)*((1ULL<<bits)-1);
    if (z < 0) z = 0;
    double mx = (double)((1ULL<<bits)-1);
    if (z > mx) z = mx;
    return (uint64_t)llround(z);
}
static uint64_t part1by1(uint64_t x) {
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x001f00000000ffffULL;
    x = (x | (x << 16)) & 0x001f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}
static vector<int> morton_order(const Instance& inst) {
    int n = (int)inst.p.size();
    double minx = inst.p[0].x, maxx = minx, miny = inst.p[0].y, maxy = miny;
    for (auto &p : inst.p) { minx=min(minx,p.x); maxx=max(maxx,p.x); miny=min(miny,p.y); maxy=max(maxy,p.y); }
    vector<pair<uint64_t,int>> v; v.reserve(n);
    for (int i=0;i<n;i++) {
        uint64_t x = norm_key(inst.p[i].x, minx, maxx), y = norm_key(inst.p[i].y, miny, maxy);
        v.push_back({part1by1(x) | (part1by1(y) << 1), i});
    }
    stable_sort(v.begin(), v.end());
    vector<int> o; o.reserve(n); for (auto &kv : v) o.push_back(kv.second);
    return o;
}
static vector<int> grid_serpentine(const Instance& inst) {
    int n = (int)inst.p.size();
    int g = max(2, (int)sqrt((double)n));
    double minx = inst.p[0].x, maxx = minx, miny = inst.p[0].y, maxy = miny;
    for (auto &p : inst.p) { minx=min(minx,p.x); maxx=max(maxx,p.x); miny=min(miny,p.y); maxy=max(maxy,p.y); }
    vector<vector<int>> bins(g);
    for (int i=0;i<n;i++) {
        int xb = (maxx > minx) ? min(g-1, (int)((inst.p[i].x-minx)/(maxx-minx)*g)) : 0;
        bins[xb].push_back(i);
    }
    vector<int> out; out.reserve(n);
    for (int b=0;b<g;b++) {
        auto &idx = bins[b];
        stable_sort(idx.begin(), idx.end(), [&](int a, int c){ return inst.p[a].y < inst.p[c].y; });
        if (b % 2) reverse(idx.begin(), idx.end());
        out.insert(out.end(), idx.begin(), idx.end());
    }
    return out;
}
static vector<int> pca_sweep(const Instance& inst) {
    int n = (int)inst.p.size();
    double mx=0,my=0;
    for (auto &p: inst.p) { mx += p.x; my += p.y; }
    mx /= n; my /= n;
    double sxx=0, syy=0, sxy=0;
    for (auto &p: inst.p) { double x=p.x-mx, y=p.y-my; sxx += x*x; syy += y*y; sxy += x*y; }
    double tr = sxx + syy, det = sxx*syy - sxy*sxy;
    double disc = max(0.0, tr*tr/4 - det);
    double lambda = tr/2 + sqrt(disc);
    double vx = sxy, vy = lambda - sxx;
    if (fabs(vx) + fabs(vy) < 1e-12) { vx = 1; vy = 0; }
    vector<pair<double,int>> a; a.reserve(n);
    for (int i=0;i<n;i++) a.push_back({(inst.p[i].x-mx)*vx + (inst.p[i].y-my)*vy, i});
    stable_sort(a.begin(), a.end());
    vector<int> o; o.reserve(n); for (auto &kv : a) o.push_back(kv.second);
    return o;
}

static vector<int> nearest_neighbor_full(const Instance& inst, int start, const Deadline& dl) {
    int n = (int)inst.p.size();
    vector<char> visited(n, 0);
    vector<int> tour; tour.reserve(n);
    int cur = ((start % n) + n) % n;
    visited[cur] = 1;
    tour.push_back(cur);
    while ((int)tour.size() < n) {
        dl.check();
        long long best = numeric_limits<long long>::max();
        int best_city = -1;
        for (int city = 0; city < n; city++) {
            if (visited[city]) continue;
            if ((city & 16383) == 0) dl.check();
            long long c = edge_cost(inst, cur, city);
            if (c < best) { best = c; best_city = city; }
        }
        if (best_city < 0) throw runtime_error("nearest_neighbor_full failed");
        cur = best_city;
        visited[cur] = 1;
        tour.push_back(cur);
    }
    return tour;
}

static void bounded_window_2opt(const Instance& inst, vector<int>& tour, int rounds, int window, const Deadline& dl) {
    int n = (int)tour.size();
    for (int r=0;r<rounds;r++) {
        dl.check();
        for (int i=0;i<n;i++) {
            if ((i & 4095) == 0) dl.check();
            int jmax = min(n-1, i + window);
            for (int j=i+2;j<=jmax;j++) {
                int a=tour[i], b=tour[(i+1)%n], c=tour[j], d=tour[(j+1)%n];
                if (edge_cost(inst,a,c) + edge_cost(inst,b,d) < edge_cost(inst,a,b) + edge_cost(inst,c,d)) {
                    reverse(tour.begin()+i+1, tour.begin()+j+1);
                }
            }
        }
    }
}

static vector<int> convex_hull_python(const Instance& inst, const Deadline& dl) {
    // Direct translation of the generated _convex_hull method. It is Jarvis march despite the Python comment.
    int n = (int)inst.p.size();
    vector<int> hull;
    int l = 0;
    for (int i=1; i<n; i++) if (inst.p[i].x < inst.p[l].x) l = i;
    int p = l;
    while (true) {
        dl.check();
        hull.push_back(p);
        int q = (p + 1) % n;
        for (int i=0; i<n; i++) {
            if ((i & 16383) == 0) dl.check();
            const Point &pp = inst.p[p], &qq = inst.p[i], &rr = inst.p[q];
            double val = (qq.y - pp.y) * (rr.x - qq.x) - (qq.x - pp.x) * (rr.y - qq.y);
            int orientation = (val == 0.0) ? 0 : (val > 0.0 ? 1 : 2);
            if (orientation == 2) q = i;
        }
        p = q;
        if (p == l) break;
    }
    return hull;
}

static void cleanup_variant_1(const Instance& inst, vector<int>& tour, int rounds, const Deadline& dl) {
    int n = (int)tour.size();
    for (int r=0; r<rounds; r++) {
        dl.check();
        bool broke_outer = false;
        for (int i=0; i<n; i++) {
            if ((i & 1023) == 0) dl.check();
            for (int j=i+2; j<n; j++) {
                if ((j & 4095) == 0) dl.check();
                long long cost = edge_cost(inst, tour[(i-1+n)%n], tour[j-1])
                               + edge_cost(inst, tour[i], tour[j])
                               - edge_cost(inst, tour[(i-1+n)%n], tour[i])
                               - edge_cost(inst, tour[j-1], tour[j]);
                if (cost < 0) {
                    reverse(tour.begin()+i, tour.begin()+j); // Python tour[i:j] reversed, j exclusive.
                    broke_outer = true;
                    break;
                }
            }
            if (broke_outer) break;
        }
    }
}

static void cleanup_variant_2(const Instance& inst, vector<int>& tour, int rounds, const Deadline& dl) {
    int n = (int)tour.size();
    for (int r=0; r<rounds; r++) {
        dl.check();
        bool broke_outer = false;
        for (int i=0; i<n; i++) {
            if ((i & 1023) == 0) dl.check();
            for (int j=i+2; j<n; j++) {
                if ((j & 4095) == 0) dl.check();
                long long cost = edge_cost(inst, tour[(i-1+n)%n], tour[j])
                               + edge_cost(inst, tour[i], tour[j-1])
                               - edge_cost(inst, tour[(i-1+n)%n], tour[i])
                               - edge_cost(inst, tour[j-1], tour[j]);
                if (cost < 0) {
                    reverse(tour.begin()+i, tour.begin()+j);
                    broke_outer = true;
                    break;
                }
            }
            if (broke_outer) break;
        }
    }
}

static vector<int> convex_outside_in_exact(const Instance& inst, int clean1, int clean2, const Deadline& dl) {
    int n = (int)inst.p.size();
    vector<int> tour = convex_hull_python(inst, dl);
    vector<char> in_tour(n, 0);
    for (int x : tour) in_tour[x] = 1;
    vector<int> interior;
    interior.reserve(n - tour.size());
    for (int i=0; i<n; i++) if (!in_tour[i]) interior.push_back(i);
    for (int city : interior) {
        dl.check();
        int best_idx = 0;
        long long best_cost = numeric_limits<long long>::max();
        int m = (int)tour.size();
        for (int i=0; i<m; i++) {
            if ((i & 4095) == 0) dl.check();
            long long cost = edge_cost(inst, tour[(i-1+m)%m], city)
                           + edge_cost(inst, city, tour[i])
                           - edge_cost(inst, tour[(i-1+m)%m], tour[i]);
            if (cost < best_cost) { best_cost = cost; best_idx = i; }
        }
        tour.insert(tour.begin()+best_idx, city);
    }
    if (clean1 > 0) cleanup_variant_1(inst, tour, clean1, dl);
    if (clean2 > 0) cleanup_variant_2(inst, tour, clean2, dl);
    return tour;
}

// -----------------------------------------------------------------------------
// Direct C++ translations of the selected distance-only LLM heuristics.
// No complexity-changing fallbacks, no replacement by unrelated scalable methods.
// -----------------------------------------------------------------------------

static vector<int> H02_normal_raw_nn2opt(const Instance& inst, RNG& rng, const Deadline& dl) {
    int n = (int)inst.p.size();
    vector<int> tour;
    vector<char> visited(n, 0);
    int start = rng.randint(n);
    tour.push_back(start);
    visited[start] = 1;
    while ((int)tour.size() < n) {
        dl.check();
        int last = tour.back();
        long long best = numeric_limits<long long>::max();
        int nearest_city = -1;
        for (int city=0; city<n; city++) {
            if (visited[city]) continue;
            if ((city & 16383) == 0) dl.check();
            long long d = edge_cost(inst, last, city);
            if (d < best) { best = d; nearest_city = city; }
        }
        tour.push_back(nearest_city);
        visited[nearest_city] = 1;
    }
    bool improved = true;
    while (improved) {
        dl.check();
        improved = false;
        for (int i=0; i<(int)tour.size()-1; i++) {
            if ((i & 127) == 0) dl.check();
            for (int j=i+1; j<(int)tour.size(); j++) {
                long long old_cost = edge_cost(inst, tour[i], tour[i+1]) + edge_cost(inst, tour[j], tour[(j+1)%n]);
                long long new_cost = edge_cost(inst, tour[i], tour[j]) + edge_cost(inst, tour[i+1], tour[(j+1)%n]);
                if (new_cost < old_cost) {
                    reverse(tour.begin()+i+1, tour.begin()+j+1); // Python i+1:j+1
                    improved = true;
                }
            }
        }
    }
    return tour;
}

static vector<int> H03_grid_exact(const Instance& inst, RNG& rng, const Deadline& dl) {
    int n = (int)inst.p.size();
    double min_x=inst.p[0].x, max_x=min_x, min_y=inst.p[0].y, max_y=min_y;
    for (auto &pt : inst.p) { min_x=min(min_x,pt.x); max_x=max(max_x,pt.x); min_y=min(min_y,pt.y); max_y=max(max_y,pt.y); }
    int num_cells = (int)sqrt((double)n);
    double cell_size_x = (max_x - min_x) / num_cells;
    double cell_size_y = (max_y - min_y) / num_cells;
    map<pair<int,int>, vector<int>> grid;
    for (int i=0; i<n; i++) {
        int cell_x = min((int)((inst.p[i].x - min_x) / cell_size_x), num_cells - 1);
        int cell_y = min((int)((inst.p[i].y - min_y) / cell_size_y), num_cells - 1);
        grid[{cell_x, cell_y}].push_back(i);
    }
    vector<pair<int,int>> ordered_cells;
    for (auto &kv : grid) ordered_cells.push_back(kv.first);
    stable_sort(ordered_cells.begin(), ordered_cells.end(), [&](auto a, auto b){
        double aa = atan2(a.second - (num_cells - 1)/2.0, a.first - (num_cells - 1)/2.0);
        double bb = atan2(b.second - (num_cells - 1)/2.0, b.first - (num_cells - 1)/2.0);
        return aa < bb;
    });
    vector<int> tour;
    vector<char> visited(n, 0);
    for (auto cell : ordered_cells) {
        dl.check();
        int nearest_city = -1;
        long long min_distance = numeric_limits<long long>::max();
        for (int city : grid[cell]) {
            if (visited[city]) continue;
            long long distance = 0;
            if (!tour.empty()) distance = edge_cost(inst, tour.back(), city);
            // Python calls rng.integers(problem.n) only in the conditional expression when tour is non-empty? Actually ternary selects tour[-1] if tour else rng.
            if (distance < min_distance) { min_distance = distance; nearest_city = city; }
        }
        if (nearest_city >= 0) { tour.push_back(nearest_city); visited[nearest_city] = 1; }
    }
    for (int i=0; i<n; i++) if (!visited[i]) {
        dl.check();
        int nearest_city_idx = -1;
        long long min_distance = numeric_limits<long long>::max();
        for (int j=0; j<(int)tour.size(); j++) {
            if ((j & 4095) == 0) dl.check();
            long long distance = edge_cost(inst, i, tour[j]);
            if (distance < min_distance) { min_distance = distance; nearest_city_idx = j; }
        }
        tour.insert(tour.begin()+nearest_city_idx+1, i);
        visited[i] = 1;
    }
    bool improved = true;
    while (improved) {
        dl.check();
        improved = false;
        for (int i=0; i<(int)tour.size()-1; i++) {
            if ((i & 127) == 0) dl.check();
            for (int j=i+1; j<(int)tour.size(); j++) {
                long long lhs = edge_cost(inst, tour[i], tour[j]) + edge_cost(inst, tour[(i+1)%n], tour[(j+1)%n]);
                long long rhs = edge_cost(inst, tour[i], tour[(i+1)%n]) + edge_cost(inst, tour[j], tour[(j+1)%n]);
                if (lhs < rhs) {
                    reverse(tour.begin()+i+1, tour.begin()+j+1);
                    improved = true;
                }
            }
        }
    }
    return tour;
}

static vector<int> H05_voronoi_exact(const Instance& inst, RNG& rng, const Deadline& dl) {
    int n = (int)inst.p.size();
    int num_seeds = max(2, (int)sqrt((double)n));
    vector<int> seeds_idx(num_seeds);
    for (int i=0; i<num_seeds; i++) seeds_idx[i] = rng.randint(n);
    vector<int> assignments(n, 0);
    for (int city=0; city<n; city++) {
        if ((city & 1023) == 0) dl.check();
        double best = numeric_limits<double>::infinity();
        int best_seed = 0;
        for (int s=0; s<num_seeds; s++) {
            double d = euclidean_norm(inst, city, seeds_idx[s]);
            if (d < best) { best = d; best_seed = s; }
        }
        assignments[city] = best_seed;
    }
    vector<vector<int>> region_tours;
    for (int s=0; s<num_seeds; s++) {
        vector<int> region_cities;
        for (int c=0; c<n; c++) if (assignments[c] == s) region_cities.push_back(c);
        if (region_cities.empty()) continue;
        vector<int> rt;
        rt.push_back(region_cities[0]);
        set<int> unvisited(region_cities.begin()+1, region_cities.end());
        while (!unvisited.empty()) {
            dl.check();
            int current_city = rt.back();
            long long best = numeric_limits<long long>::max();
            int next_city = -1;
            for (int x : unvisited) {
                long long d = edge_cost(inst, current_city, x);
                if (d < best) { best = d; next_city = x; }
            }
            rt.push_back(next_city);
            unvisited.erase(next_city);
        }
        region_tours.push_back(move(rt));
    }
    vector<int> tour;
    int last_region_city = -1;
    for (int i=0; i<(int)region_tours.size(); i++) {
        dl.check();
        if (i == 0) {
            tour.insert(tour.end(), region_tours[i].begin(), region_tours[i].end());
            last_region_city = region_tours[i].back();
        } else {
            int best_city = -1, best_idx = -1;
            long long best_cost = numeric_limits<long long>::max();
            for (int j=0; j<(int)region_tours[i].size(); j++) {
                long long cost = edge_cost(inst, last_region_city, region_tours[i][j]);
                if (cost < best_cost) { best_cost = cost; best_city = region_tours[i][j]; best_idx = j; }
            }
            tour.push_back(best_city);
            for (int j=0; j<(int)region_tours[i].size(); j++) if (j != best_idx) tour.push_back(region_tours[i][j]);
            last_region_city = region_tours[i].back();
        }
    }
    vector<char> in_tour(n, 0);
    for (int x : tour) in_tour[x] = 1;
    for (int city=0; city<n; city++) if (!in_tour[city]) {
        dl.check();
        long long best = numeric_limits<long long>::max();
        int nearest_idx = -1;
        for (int idx=0; idx<(int)tour.size(); idx++) {
            long long d = edge_cost(inst, tour[idx], city);
            if (d < best) { best = d; nearest_idx = idx; }
        }
        tour.insert(tour.begin()+nearest_idx+1, city);
    }
    int rounds = max(10, (int)log((double)n));
    for (int r=0; r<rounds; r++) {
        dl.check();
        bool improved = false;
        for (int i=0; i<(int)tour.size()-1; i++) {
            if ((i & 127) == 0) dl.check();
            for (int j=i+2; j<(int)tour.size(); j++) {
                long long cost1 = edge_cost(inst, tour[i], tour[i+1]) + edge_cost(inst, tour[j-1], tour[j]);
                long long cost2 = edge_cost(inst, tour[i], tour[j-1]) + edge_cost(inst, tour[i+1], tour[j]);
                if (cost2 < cost1) {
                    reverse(tour.begin()+i+1, tour.begin()+j); // Python i+1:j, j exclusive.
                    improved = true;
                }
            }
        }
        if (!improved) break;
    }
    return tour;
}

static vector<int> H07_region_endpoint_exact(const Instance& inst, RNG& rng, const Deadline& dl) {
    int n = (int)inst.p.size();
    vector<int> tour;
    set<int> open_endpoints;
    for (int i=0;i<n;i++) open_endpoints.insert(i);
    int current_city = rng.randint(n);
    tour.push_back(current_city);
    open_endpoints.erase(current_city);
    vector<vector<int>> fragments;
    fragments.push_back({current_city});
    while ((int)tour.size() < n) {
        dl.check();
        vector<int> closest_cities;
        for (auto &fragment : fragments) {
            long long min_distance = numeric_limits<long long>::max();
            int closest_city = -1;
            for (int city : open_endpoints) {
                long long distance = edge_cost(inst, fragment.back(), city);
                if (distance < min_distance) { min_distance = distance; closest_city = city; }
            }
            closest_cities.push_back(closest_city);
        }
        long long min_distance = numeric_limits<long long>::max();
        int next_fragment_index = -1;
        int next_city = -1;
        for (int i=0; i<(int)fragments.size(); i++) {
            int closest_city = closest_cities[i];
            long long distance = edge_cost(inst, fragments[i].back(), closest_city);
            if (distance < min_distance) { min_distance = distance; next_fragment_index = i; next_city = closest_city; }
        }
        fragments[next_fragment_index].push_back(next_city);
        open_endpoints.erase(next_city);
        if ((int)fragments[next_fragment_index].size() > 1) {
            bool next_is_fragment_start = false;
            for (auto &fragment : fragments) if (!fragment.empty() && next_city == fragment[0]) { next_is_fragment_start = true; break; }
            if (next_is_fragment_start) fragments.erase(fragments.begin()+next_fragment_index);
        }
        tour.clear();
        for (auto &fragment : fragments) for (int city : fragment) tour.push_back(city);
        if (fragments.size() > 1) {
            int min_length = numeric_limits<int>::max(), min_fragment_index = -1;
            for (int i=0; i<(int)fragments.size(); i++) if ((int)fragments[i].size() < min_length) { min_length = fragments[i].size(); min_fragment_index = i; }
            int max_length = 0, max_fragment_index = -1;
            for (int i=0; i<(int)fragments.size(); i++) if ((int)fragments[i].size() > max_length) { max_length = fragments[i].size(); max_fragment_index = i; }
            if (min_fragment_index != max_fragment_index && min_fragment_index >= 0 && max_fragment_index >= 0) {
                fragments[max_fragment_index].insert(fragments[max_fragment_index].end(), fragments[min_fragment_index].begin(), fragments[min_fragment_index].end());
                fragments.erase(fragments.begin()+min_fragment_index);
            }
        }
    }
    return tour;
}

static vector<int> H08_geostabilizer_exact(const Instance& inst, int start_node, const Deadline& dl) {
    int n = (int)inst.p.size();
    vector<int> tour;
    vector<char> visited(n, 0);
    start_node = ((start_node % n) + n) % n;
    tour.push_back(start_node);
    visited[start_node] = 1;
    for (int step=0; step<n-1; step++) {
        dl.check();
        long long min_dist = numeric_limits<long long>::max();
        int next_node = -1;
        for (int i=0; i<n; i++) {
            if (visited[i]) continue;
            if ((i & 16383) == 0) dl.check();
            long long dist = edge_cost(inst, tour.back(), i);
            if (dist < min_dist) {
                min_dist = dist;
                next_node = i;
            } else if (dist == min_dist) {
                // Direct translation of the odd Python tie-break:
                // centroid = np.mean([tour], axis=0); compare norm(centroid - i).
                // Since [tour] is a 1-row array, centroid is the vector of current tour indices.
                long double di = 0.0L, dn = 0.0L;
                for (int v : tour) { long double ai = (long double)v - i; di += ai*ai; long double an = (long double)v - next_node; dn += an*an; }
                if (sqrt((double)di) < sqrt((double)dn)) next_node = i;
            }
        }
        if (next_node < 0) throw runtime_error("geostabilizer failed to find next node");
        tour.push_back(next_node);
        visited[next_node] = 1;
    }
    return tour;
}

static vector<int> H09_mst_diagnostic_exact(const Instance& inst, RNG& rng, const Deadline& dl) {
    int n = (int)inst.p.size();
    set<int> visited;
    int current_city = 0;
    visited.insert(current_city);
    vector<int> tour;
    tour.push_back(current_city);
    unordered_map<int, vector<int>> mst;
    vector<int> cities(n); iota(cities.begin(), cities.end(), 0);
    while ((int)visited.size() < n) {
        dl.check();
        long long min_cost = numeric_limits<long long>::max();
        int next_city = -1;
        for (int city : cities) {
            if (visited.count(city)) continue;
            long long cost = edge_cost(inst, current_city, city);
            if (cost < min_cost) { min_cost = cost; next_city = city; }
        }
        tour.push_back(next_city);
        visited.insert(next_city);
        current_city = next_city;
        mst[current_city].push_back(tour[tour.size()-2]);
        mst[tour[tour.size()-2]].push_back(current_city);
    }
    current_city = tour[0];
    vector<int> hamiltonian_tour;
    hamiltonian_tour.push_back(current_city);
    vector<char> in_ham(n, 0); in_ham[current_city] = 1;
    while ((int)hamiltonian_tour.size() < n) {
        dl.check();
        vector<int> unvisited_neighbors;
        for (int neighbor : mst[current_city]) if (!in_ham[neighbor]) unvisited_neighbors.push_back(neighbor);
        int next_city = -1;
        if (!unvisited_neighbors.empty()) {
            long long best = numeric_limits<long long>::max();
            for (int neighbor : unvisited_neighbors) {
                long long d = edge_cost(inst, current_city, neighbor);
                if (d < best) { best = d; next_city = neighbor; }
            }
        } else {
            long long best = numeric_limits<long long>::max();
            for (int city=0; city<n; city++) if (!in_ham[city]) {
                long long d = edge_cost(inst, current_city, city);
                if (d < best) { best = d; next_city = city; }
            }
        }
        hamiltonian_tour.push_back(next_city);
        in_ham[next_city] = 1;
        current_city = next_city;
    }
    for (int it=0; it<(int)(n * 0.2); it++) {
        dl.check();
        int i = rng.randint_range(1, n-1); // [1, n-1)
        int j = rng.randint_range(i+1, n); // [i+1, n)
        if (edge_cost(inst, hamiltonian_tour[i-1], hamiltonian_tour[j]) + edge_cost(inst, hamiltonian_tour[i], hamiltonian_tour[j-1])
            < edge_cost(inst, hamiltonian_tour[i-1], hamiltonian_tour[i]) + edge_cost(inst, hamiltonian_tour[j-1], hamiltonian_tour[j])) {
            reverse(hamiltonian_tour.begin()+i, hamiltonian_tour.begin()+j); // Python i:j
        }
    }
    return hamiltonian_tour;
}


static vector<int> H12_D1_nn_constructive_only(const Instance& inst, RNG& rng, const Deadline& dl) {
    // Direct translation of D1_nn_constructive_only/heuristic.py:
    // - perform min(10, n) random nearest-neighbor starts
    // - keep the tour with the lowest objective value
    int n = (int)inst.p.size();
    vector<int> best_tour;
    double best_cost = numeric_limits<double>::infinity();
    int starts = min(10, n);
    for (int r=0; r<starts; r++) {
        dl.check();
        int current_city = rng.randint(n);
        vector<int> tour;
        tour.reserve(n);
        vector<char> visited(n, 0);
        tour.push_back(current_city);
        visited[current_city] = 1;
        while ((int)tour.size() < n) {
            dl.check();
            long long best = numeric_limits<long long>::max();
            int nearest_city = -1;
            for (int city=0; city<n; city++) {
                if (visited[city]) continue;
                if ((city & 16383) == 0) dl.check();
                long long d = edge_cost(inst, current_city, city);
                if (d < best) { best = d; nearest_city = city; }
            }
            if (nearest_city < 0) throw runtime_error("D1 nearest-neighbor failed");
            tour.push_back(nearest_city);
            current_city = nearest_city;
            visited[current_city] = 1;
        }
        double c = tour_cost(inst, tour);
        if (c < best_cost) {
            best_cost = c;
            best_tour = move(tour);
        }
    }
    return best_tour;
}

static vector<int> method_tour(const string& method, const Instance& inst, uint64_t seed, double timeout_s) {
    Deadline dl(timeout_s);
    RNG rng(seed);
    int n = (int)inst.p.size();
    // Selected LLM heuristics, direct structural translations.
    if (method == "02_normal_raw_nn2opt_best_101102_iter003") return H02_normal_raw_nn2opt(inst, rng, dl);
    if (method == "03_family_focus_grid_best_100159_iter072") return H03_grid_exact(inst, rng, dl);
    if (method == "04_family_focus_convex_faithful_095803_iter031") return convex_outside_in_exact(inst, 50, 20, dl);
    if (method == "05_family_focus_voronoi_best_100159_iter037") return H05_voronoi_exact(inst, rng, dl);
    if (method == "07_family_focus_region_endpoint_fast_100159_iter177") return H07_region_endpoint_exact(inst, rng, dl);
    if (method == "08_expo_distance_only_geostabilizer_399e") return H08_geostabilizer_exact(inst, (int)(seed % n), dl);
    if (method == "09_family_focus_mst_diagnostic_100159_iter007") return H09_mst_diagnostic_exact(inst, rng, dl);
    if (method == "10_family_focus_fast_convex_095803_iter026") return convex_outside_in_exact(inst, 5, 2, dl);
    if (method == "11_family_focus_convex_constructive_095803_iter021") return convex_outside_in_exact(inst, 0, 0, dl);
    if (method == "12_D1_nn_constructive_only") return H12_D1_nn_constructive_only(inst, rng, dl);
    if (method == "13_D2a_convex_hull_outside_in_with_cleanup") return convex_outside_in_exact(inst, 3, 0, dl);

    // C++ external baselines. These are not translations of selected LLM code.
    if (method == "01_kdtree_nearest_neighbor_fixed_start") return nearest_neighbor_full(inst, 0, dl);
    if (method == "02_kdtree_nearest_neighbor_multistart") {
        vector<int> starts = {0, n/4, n/2, (3*n)/4};
        vector<int> best; double bc = numeric_limits<double>::infinity();
        for (int st : starts) { auto t = nearest_neighbor_full(inst, st, dl); double c = tour_cost(inst, t); if (c < bc) { bc = c; best = move(t); } }
        return best;
    }
    if (method == "03_x_axis_sweep") return x_axis_sweep(inst);
    if (method == "04_pca_sweep") return pca_sweep(inst);
    if (method == "05_angular_sweep") return angular_sweep(inst);
    if (method == "06_morton_z_order") return morton_order(inst);
    if (method == "07_grid_serpentine") return grid_serpentine(inst);
    if (method == "08_morton_bounded_local_2opt") { auto tour = morton_order(inst); bounded_window_2opt(inst, tour, 2, 48, dl); return tour; }
    throw runtime_error("unknown method: " + method);
}

static fs::path find_tsp(const fs::path& root, const string& name) {
    vector<fs::path> c = {root/(name + ".tsp"), root/(name + ".TSP"), root/name/(name + ".tsp"), root/name/(name + ".TSP")};
    for (auto &p : c) if (fs::exists(p)) return p;
    throw runtime_error("missing tsp for " + name);
}

int main(int argc, char** argv) {
    unordered_map<string,string> arg;
    for (int i=1; i<argc; i++) {
        string a = argv[i];
        if (a.rfind("--", 0) == 0 && i+1 < argc) arg[a.substr(2)] = argv[++i];
    }
    string job = arg["job"], kind = arg["kind"], method = arg["method"];
    string signal = arg.count("signal") ? arg["signal"] : "distance_only";
    fs::path inst_root = arg["instance-root"], opt_csv = arg["optima-csv"], out_dir = arg["out-dir"];
    int reps = stoi(arg.count("reps") ? arg["reps"] : "50");
    uint64_t global_seed = stoull(arg.count("global-seed") ? arg["global-seed"] : "12345");
    double timeout_s = stod(arg.count("timeout-s") ? arg["timeout-s"] : "900");
    vector<string> wanted = split_csv(arg["instances"]);
    fs::create_directories(out_dir);

    vector<OptRow> opt = read_opt_csv(opt_csv);
    vector<Instance> instances;
    for (auto &r : opt) {
        if (!wanted.empty() && !contains(wanted, r.instance)) continue;
        instances.push_back(read_tsp(find_tsp(inst_root, r.instance), r.instance, r.split, r.opt));
    }

    fs::path raw = out_dir/"raw_results.csv";
    bool exists = fs::exists(raw);
    ofstream csv(raw, ios::app);
    if (!exists) {
        csv << "signal_category,heuristic_id,heuristic_label,code_path,instance_name,split,n,rep,seed,objective_value,reference_value,gap_ref_pct,runtime_s,status,error_type,error_message,hostname\n";
    }
    char host[256]; gethostname(host, sizeof(host));

    for (int rep=1; rep<=reps; rep++) {
        for (const auto &inst : instances) {
            uint64_t seed = global_seed + rep*1000003ULL + std::hash<string>{}(inst.name) + std::hash<string>{}(method);
            auto t0 = chrono::steady_clock::now();
            string status="ok", et="", em="";
            double obj = numeric_limits<double>::quiet_NaN();
            double gap = numeric_limits<double>::quiet_NaN();
            try {
                auto tour = method_tour(method, inst, seed, timeout_s);
                validate_tour(tour, (int)inst.p.size());
                obj = tour_cost(inst, tour);
                gap = 100.0 * (obj - inst.opt) / inst.opt;
            } catch (const exception& e) {
                string msg = e.what();
                if (msg.find("timeout") != string::npos) { status = "timeout"; et = "Timeout"; }
                else { status = "error"; et = "RuntimeError"; }
                em = msg;
            }
            double rt = chrono::duration<double>(chrono::steady_clock::now()-t0).count();
            csv << csv_escape(signal) << "," << csv_escape(job) << "," << csv_escape(method)
                << ",server_eval/tsp_cpp_distance_eval.cpp," << csv_escape(inst.name) << "," << csv_escape(inst.split)
                << "," << inst.p.size() << "," << rep << "," << seed << ",";
            if (isfinite(obj)) csv << fixed << setprecision(6) << obj;
            csv << "," << fixed << setprecision(6) << inst.opt << ",";
            if (isfinite(gap)) csv << fixed << setprecision(9) << gap;
            csv << "," << fixed << setprecision(6) << rt << "," << status << "," << et << "," << csv_escape(em) << "," << host << "\n";
            csv.flush();
        }
    }
    ofstream sum(out_dir/"summary_by_heuristic.csv");
    sum << "signal_category,heuristic_id,heuristic_label,total_runs,ok_runs,error_runs,timeout_runs,runtime_note\n";
    sum << signal << "," << job << "," << method << ",see_raw_results,see_raw_results,see_raw_results,see_raw_results,cpp_direct_translation_no_fallback\n";
    return 0;
}
