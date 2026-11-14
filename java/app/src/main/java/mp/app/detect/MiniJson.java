package mp.app.detect;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Tiny JSON reader (objects, arrays, strings, numbers, booleans, null). */
public final class MiniJson {
    private final String s;
    private int p = 0;

    private MiniJson(String s) { this.s = s; }

    public static Object parse(String text) {
        MiniJson j = new MiniJson(text);
        Object v = j.value();
        j.ws();
        if (j.p != j.s.length()) throw new IllegalArgumentException("Trailing data at " + j.p);
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asObj(Object o) { return (Map<String, Object>) o; }

    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) { return (List<Object>) o; }

    public static double num(Object o) { return ((Number) o).doubleValue(); }

    private void ws() { while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++; }

    private IllegalArgumentException err() { return new IllegalArgumentException("JSON syntax error at " + p); }

    private void expect(char c) { if (s.charAt(p) != c) throw err(); p++; }

    private Object value() {
        ws();
        char c = s.charAt(p);
        if (c == '{') return obj();
        if (c == '[') return arr();
        if (c == '"') return str();
        if (s.startsWith("true", p)) { p += 4; return Boolean.TRUE; }
        if (s.startsWith("false", p)) { p += 5; return Boolean.FALSE; }
        if (s.startsWith("null", p)) { p += 4; return null; }
        int q = p;
        while (q < s.length() && "+-0123456789.eE".indexOf(s.charAt(q)) >= 0) q++;
        if (q == p) throw err();
        double d = Double.parseDouble(s.substring(p, q));
        p = q;
        return d;
    }

    private Map<String, Object> obj() {
        Map<String, Object> m = new LinkedHashMap<>();
        expect('{'); ws();
        if (s.charAt(p) == '}') { p++; return m; }
        while (true) {
            ws(); String k = str(); ws(); expect(':');
            m.put(k, value()); ws();
            char c = s.charAt(p++);
            if (c == '}') return m;
            if (c != ',') throw err();
        }
    }

    private List<Object> arr() {
        List<Object> l = new ArrayList<>();
        expect('['); ws();
        if (s.charAt(p) == ']') { p++; return l; }
        while (true) {
            l.add(value()); ws();
            char c = s.charAt(p++);
            if (c == ']') return l;
            if (c != ',') throw err();
        }
    }

    private String str() {
        expect('"');
        StringBuilder b = new StringBuilder();
        while (true) {
            char c = s.charAt(p++);
            if (c == '"') return b.toString();
            if (c != '\\') { b.append(c); continue; }
            char e = s.charAt(p++);
            switch (e) {
                case 'n' -> b.append('\n');
                case 't' -> b.append('\t');
                case 'r' -> b.append('\r');
                case 'b' -> b.append('\b');
                case 'f' -> b.append('\f');
                case 'u' -> { b.append((char) Integer.parseInt(s.substring(p, p + 4), 16)); p += 4; }
                default -> b.append(e);
            }
        }
    }
}
