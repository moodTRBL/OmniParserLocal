// package: https://www.npmjs.com/package/functional-red-black-tree
class PointLocator {
    constructor(delaunay) {
      const { verts } = delaunay;
  
      this.lx = 0;
      this.rx = 0;
      this.compareMode = 'sort';
  
      this.xs = this.initializeXs(verts);
      this.trees = this.initializeTrees(this.xs, delaunay);
    }
  
    initializeXs(verts) {
      verts = verts.map(({ pos, idx }) => ({ pos, origin: idx }))
        .sort((a, b) => {
          if (a.pos.cmp(b.pos) !== 0) {
            return a.pos.cmp(b.pos);
          }
          return a.origin - b.origin;
        })
        .map(({ pos, origin }, idx) => ({ x: pos.x, origin, idx }));
      verts = _.uniqBy(verts, 'x');
  
      return verts;
    }
  
    initializeTrees(xs, delaunay) {
      let tree = createRBTree((a, b) => this.compareLine(a, b));
      const trees = [tree];
  
      const verts = delaunay.verts.map((vert) => ({
        ...vert,
        xidx: bounds.eq(xs, { x: vert.pos.x }, (a, b) => a.x - b.x)
      }))
        .sort((a, b) => a.xidx - b.xidx);
  
      for (let i = 0, p = 0; i < xs.length; i++) {
        const vs = [];
        while (p < verts.length && verts[p].xidx === i) {
          vs.push(verts[p++]);
        }
  
        const [inserts, removes] = [[], []];
        for (const { idx } of vs) {
          for (const edge of delaunay.adjust(idx)) {
            const to_x = bounds.eq(xs, { x: edge.to.x }, (a, b) => a.x - b.x);
            if (to_x < i) {
              removes.push({ ...edge, to_x });
            } else {
              inserts.push({ ...edge, to_x });
            }
          }
        }
  
        const lines = [
          ...removes,
          ...inserts,
        ].map((line) => {
          let [l, r] = [line.from, line.to];
          if (l.cmp(r) > 0) {
            [l, r] = [r, l];
          }
  
          return {
            l,
            r,
            triangle: line.triangle,
            to_x: line.to_x,
            idx: line.idx,
          };
        })
          .filter((line) => line.l.x !== line.r.x)
          .sort((a, b) => a.to_x - b.to_x);
  
        for (const line of lines) {
          if (line.to_x < i) {
            this.setCompareLine(i - 1);
            tree = tree.remove(line);
          } else {
            this.setCompareLine(i);
            tree = tree.insert(line);
          }
        }
  
        trees.push(tree);
      }
  
      return trees;
    }
  
    setCompareLine(slapIdx) {
      const { xs } = this;
      this.lx = xs[slapIdx].x;
      this.rx = slapIdx + 1 < xs.length ? xs[slapIdx + 1].x : 1e6;
    }
  
    compareLineSort(a, b) {
      const { lx, rx } = this;
      const shrink = (l, r) => {
        const m = (r.y - l.y) / (r.x - l.x);
        l = new v2(lx, m * (lx - l.x) + l.y);
        r = new v2(rx, m * (rx - r.x) + r.y);
        return [l, r];
      };
      const [al, ar] = shrink(a.l, a.r);
      const [bl, br] = shrink(b.l, b.r);
  
      const ay = new v2(al.y, ar.y);
      const by = new v2(bl.y, br.y);
  
      if (ay.cmp(by) !== 0) {
        return ay.cmp(by);
      }
      return a.triangle - b.triangle;
    }
  
    compareLineFind(a, b) {
      const ccw = v2.ccw(b.l, b.r, a.l, true) + v2.ccw(b.l, b.r, a.r, true);
      if (ccw !== 0) {
        return ccw;
      }
      return a.triangle - b.triangle;
    }
  
    compareLine(a, b) {
      const { compareMode } = this;
      if (compareMode === 'sort') {
        return this.compareLineSort(a, b);
      } else {
        return this.compareLineFind(a, b);
      }
    }
  
    getSlapIdx(point) {
      return bounds.le(this.xs, { x: point.x }, (a, b) => a.x - b.x);
    }
  
    locate(point) {
      const slapIdx = Math.max(1, this.getSlapIdx(point));
      const tree = this.trees[slapIdx + 1];
  
      this.compareMode = 'find';
      const ans = tree.le({ l: point, r: point, triangle: -1 });
      this.compareMode = 'sort';
      if (!ans?.key) {
        return -1;
      }
  
      return ans.key.triangle;
    }
  }