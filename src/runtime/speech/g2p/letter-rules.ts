// Letter-to-sound rules for English words no lexicon knows, after Elovitz, Johnson, McHugh and Shore, "Automatic
// translation of English text to phonetics by means of letter-to-sound rules" (Naval Research Laboratory report
// 7948, 1976, a US government work), written here in Misaki phonemes. Each rule is "left[letters]right=phonemes";
// the first rule for the current letter whose contexts match wins. Context symbols:
//   ' ' word edge   '#' one or more vowels   '*' one or more consonants   '^' one consonant   ':' zero or more
//   consonants   '+' e, i or y   '.' a voiced consonant   '%' a suffix (er, e, es, ed, ing, ely)
//   '&' a sibilant   '@' a consonant after which u is "oo"

const RULES = `
a: [a] =ə| [are] =ɑɹ| [ar]o=əɹ|[ar]#=ɛɹ| ^[as]#=As|[a]wa=ə|[aw]=ɔ| :[any]=ɛni|[a]^+#=A|#:[ally]=əli| [al]#=əl|[again]=əɡɛn|#:[ag]e=ɪʤ|[a]^+:#=æ| :[a]^+ =A|[a]^%=A| [arr]=əɹ|[arr]=æɹ| :[ar] =ɑɹ|[ar] =əɹ|[ar]=ɑɹ|[air]=ɛɹ|[ai]=A|[ay]=A|[au]=ɔ|#:[al] =əl|#:[als] =əlz|[alk]=ɔk|[al]^=ɔl| :[able]=Abəl|[able]=əbəl|[ang]+=Anʤ|[a]=æ
b: [be]^#=bɪ|[being]=biɪŋ| [both] =bOθ| [bus]#=bɪz|[buil]=bɪl|[bb]=b|[b]=b
c: [ch]^=k|^e[ch]=k|[ch]=ʧ| s[ci]#=sI|[ci]a=ʃ|[ci]o=ʃ|[ci]en=ʃ|[c]+=s|[ck]=k|[com]%=kʌm|[cc]+=ks|[c]=k
d: #:[ded] =dɪd|.e[d] =d|#^:e[d] =t| [de]^#=dɪ| [do] =du| [does]=dʌz| [doing]=duɪŋ| [dow]=dW|[du]a=ʤu|[dd]=d|[d]=d
e: #:[e] =| :[e] =i|#[ed] =d|#:[e]d =|[ev]er=ɛv|[e]^%=i|[eri]#=iɹi|[eri]=ɛɹɪ|#:[er]#=əɹ|[er]#=ɛɹ|[er]=əɹ| [even]=ivən|#:[e]w=|@[ew]=u|[ew]=ju|[e]o=i|#:&[es] =ᵻz|#:[e]s =|#:[ely] =li|#:[ement]=mənt|[eful]=fʊl|[ee]=i|[earn]=ɜɹn| [ear]^=ɜɹ|[ead]=ɛd|#:[ea] =iə|[ea]su=ɛ|[ea]=i|[eigh]=A|[ei]=i| [eye]=I|[ey]=i|[eu]=ju|[e]=ɛ
f: [ful]=fʊl|[ff]=f|[f]=f
g: [giv]=ɡɪv| [g]i^=ɡ|[ge]t=ɡɛ|su[gges]=ɡʤɛs|[gg]=ɡ| b#[g]=ɡ|[g]+=ʤ|[great]=ɡɹAt|#[gh]=|[g]=ɡ
h: [hav]=hæv| [here]=hiɹ| [hour]=Wəɹ|[how]=hW|[h]#=h|[h]=
i: [in]=ɪn| [i] =I|[in]d=In|[ier]=iəɹ|#:r[ied] =id|[ied] =Id|[ien]=iɛn|[ie]t=Iɛ| :[i]%=I|[i]%=i|[ie]=i|[i]^+:#=ɪ|[ir]#=Iɹ|[iz]%=Iz|[is]%=Iz|[i]d%=I|+^[i]^+=ɪ|[i]t%=I|#^:[i]^+=ɪ|[i]^+=I|[ir]=ɜɹ|[igh]=I|[ild]=Ild|[ign] =In|[ign]^=In|[ign]%=In|[ique]=ik|[i]=ɪ
j: [j]=ʤ
k: [k]n=|[k]=k
l: [lo]c#=lO|l[l]=|#^:[l]%=əl|[lead]=lid|[l]=l
m: [mov]=muv|[mm]=m|[m]=m
n: e[ng]+=nʤ|[ng]r=ŋɡ|[ng]#=ŋɡ|[ngl]%=ŋɡəl|[ng]=ŋ|[nk]=ŋk| [now] =nW|[nn]=n|[n]=n
o: [of] =əv|[orough]=ɜɹO|#:[or] =əɹ|#:[ors] =əɹz|[or]=ɔɹ| [one]=wʌn|[ow]=O| [over]=Ovəɹ|[ov]=ʌv|[o]^%=O|[o]^en=O|[o]^i#=O|[ol]d=Ol|[ought]=ɔt|[ough]=ʌf| [ou]=W|h[ou]s#=W|[ous]=əs|[our]=ɔɹ|[ould]=ʊd|^[ou]^l=ʌ|[oup]=up|[ou]=W|[oy]=Y|[oing]=Oɪŋ|[oi]=Y|[oor]=ɔɹ|[ook]=ʊk|[ood]=ʊd|[oo]=u|[o]e=O|[o] =O|[oa]=O| [only]=Onli| [once]=wʌns|c[o]n=ɑ|[o]ng=ɔ| :^[o]n=ʌ|i[on]=ən|#:[on] =ən|#^[on]=ən|[o]st =O|[of]^=ɔf|[other]=ʌðəɹ|[oss] =ɔs|#^:[om]=əm|[o]=ɑ
p: [ph]=f|[peop]=pip|[pow]=pW|[put] =pʊt|[pp]=p|[p]=p
q: [quar]=kwɔɹ|[qu]=kw|[q]=k
r: [re]^#=ɹi|[rr]=ɹ|[r]=ɹ
s: [sh]=ʃ|#[sion]=ʒən|[some]=sʌm|#[sur]#=ʒəɹ|[sur]#=ʃəɹ|#[su]#=ʒu|#[ssu]#=ʃu|#[sed] =zd|#[s]#=z|[said]=sɛd|^[sion]=ʃən|[s]s=|.[s] =z|#:.e[s] =z|#^:##[s] =z|#^:#[s] =s|u[s] =s| :#[s] =z| [sch]=sk|[s]c+=|#[sm]=zəm|[s]=s
t: [the] =ðə| [this] =ðɪs| [they]=ðA| [there]=ðɛɹ|[ther]=ðəɹ|[their]=ðɛɹ| [than] =ðæn| [them] =ðɛm|[these] =ðiz| [then]=ðɛn|[through]=θɹu|[those]=ðOz|[though] =ðO| [thus]=ðʌs|[th]=θ|#:[ted] =tᵻd|s[ti]#n=ʧ|[ti]o=ʃ|[ti]a=ʃ|[tien]=ʃən|[tur]#=ʧəɹ|[tu]a=ʧu| [two]=tu|[tt]=t|[tch]=ʧ|[t]=t
u: [un]i=jun| [un]=ʌn| [upon]=əpɔn|@[ur]#=ʊɹ|[ur]#=jʊɹ|[ur]=ɜɹ|[u]^ =ʌ|[u]^^=ʌ|[uy]=I| g[u]#=|g[u]%=|g[u]#=w|#n[u]=ju|@[u]=u|[u]=ju
v: [view]=vju|[v]=v
w: [were]=wɜɹ|[wa]s=wɑ|[wa]t=wɑ|[where]=wɛɹ|[what]=wʌt|[whol]=hOl|[who]=hu|[wh]=w|[war]=wɔɹ|[wor]^=wɜɹ|[wr]=ɹ|[w]=w
x: [x]=ks
y: [young]=jʌŋ| [you]=ju| [yes]=jɛs| [y]=j|#^:[y] =i|#^:[y]i=i| :[y] =I| :[y]#=I| :[y]^+:#=ɪ| :[y]^#=I|[y]=ɪ
z: [zz]=z|[z]=z
`;

interface Rule {
  left: string;
  match: string;
  right: string;
  out: string;
}

const RULE_TABLE = new Map<string, Rule[]>();
for (const line of RULES.trim().split('\n')) {
  const [letter, body = ''] = line.split(/:(.*)/s);
  RULE_TABLE.set(letter!, body.split('|').map((r) => {
    const m = /^(.*)\[(.+)\](.*)=(.*)$/.exec(r)!;
    return { left: m[1]!, match: m[2]!, right: m[3]!, out: m[4]! };
  }));
}

const isVowel = (c: string | undefined) => !!c && 'aeiouy'.includes(c);
const isConsonant = (c: string | undefined) => !!c && /[a-z]/.test(c) && !isVowel(c);
const SUFFIXES = ['ing', 'ely', 'er', 'es', 'ed', 'e'];

/** Whether the right context pattern matches the word from position i on. */
function matchRight(pattern: string, w: string, i: number): boolean {
  if (!pattern) return true;
  const p = pattern[0]!;
  const rest = pattern.slice(1);
  const c = w[i];
  switch (p) {
    case ' ': return c === ' ' && matchRight(rest, w, i + 1);
    case '#': {
      if (!isVowel(c)) return false;
      let j = i;
      while (isVowel(w[j])) j++;
      return matchRight(rest, w, j);
    }
    case '*': {
      if (!isConsonant(c)) return false;
      let j = i;
      while (isConsonant(w[j])) j++;
      return matchRight(rest, w, j);
    }
    case ':': {
      let j = i;
      while (isConsonant(w[j])) j++;
      return matchRight(rest, w, j);
    }
    case '^': return isConsonant(c) && matchRight(rest, w, i + 1);
    case '+': return !!c && 'eiy'.includes(c) && matchRight(rest, w, i + 1);
    case '.': return !!c && 'bdvgjlmnrwz'.includes(c) && matchRight(rest, w, i + 1);
    case '&': return w.startsWith('ch', i) || w.startsWith('sh', i) ? matchRight(rest, w, i + 2) : !!c && 'scgzxj'.includes(c) && matchRight(rest, w, i + 1);
    case '@': return w.startsWith('th', i) || w.startsWith('ch', i) || w.startsWith('sh', i) ? matchRight(rest, w, i + 2) : !!c && 'tsrdlznj'.includes(c) && matchRight(rest, w, i + 1);
    case '%': return SUFFIXES.some((s) => w.startsWith(s, i) && w[i + s.length] === ' ' && matchRight(rest, w, i + s.length));
    default: return c === p && matchRight(rest, w, i + 1);
  }
}

/** Whether the left context pattern (read right to left) matches the word before position i. */
function matchLeft(pattern: string, w: string, i: number): boolean {
  if (!pattern) return true;
  const p = pattern[pattern.length - 1]!;
  const rest = pattern.slice(0, -1);
  const c = w[i - 1];
  switch (p) {
    case ' ': return c === ' ' && matchLeft(rest, w, i - 1);
    case '#': {
      if (!isVowel(c)) return false;
      let j = i - 1;
      while (isVowel(w[j - 1])) j--;
      return matchLeft(rest, w, j);
    }
    case '*': {
      if (!isConsonant(c)) return false;
      let j = i - 1;
      while (isConsonant(w[j - 1])) j--;
      return matchLeft(rest, w, j);
    }
    case ':': {
      let j = i;
      while (isConsonant(w[j - 1])) j--;
      return matchLeft(rest, w, j);
    }
    case '^': return isConsonant(c) && matchLeft(rest, w, i - 1);
    case '+': return !!c && 'eiy'.includes(c) && matchLeft(rest, w, i - 1);
    case '.': return !!c && 'bdvgjlmnrwz'.includes(c) && matchLeft(rest, w, i - 1);
    case '&': return !!c && 'scgzxjh'.includes(c) && matchLeft(rest, w, i - 1);
    case '@': return !!c && 'tsrdlznjh'.includes(c) && matchLeft(rest, w, i - 1);
    default: return c === p && matchLeft(rest, w, i - 1);
  }
}

/** Unstressed phonemes for a lower-case word of letters a–z. */
export function letterToSound(word: string): string {
  const w = ` ${word} `;
  let out = '';
  for (let i = 1; i < w.length - 1;) {
    const rule = RULE_TABLE.get(w[i]!)?.find((r) => w.startsWith(r.match, i) && matchLeft(r.left, w, i) && matchRight(r.right, w, i + r.match.length));
    if (!rule) {
      i++;
      continue;
    }
    out += rule.out;
    i += rule.match.length;
  }
  return out;
}
