namespace std __attribute__ ((_ ("default")))
{
 template<class _CharT>
 struct char_traits;
}
typedef int size_t;
namespace std __attribute__ ((_ ("default")))
{
 typedef long streamsize;
 template<typename _CharT, typename = char_traits<_CharT> >
 class basic_ostream;
 typedef basic_ostream<char> ostream;
 template<>
 struct char_traits<char>
 {
 typedef char char_type;
 static size_t
 length(char_type* __s)
 { return __builtin_strlen(__s); }
 };
 template<typename _CharT, typename _Traits>
 basic_ostream<_CharT, _Traits>&
 __ostream_insert(basic_ostream<_CharT, _Traits>& _out,
 const _CharT* _s, streamsize);
 class ios_base
 {
 public:
 class Init
 {
 public:
 Init();
 };
 };
 template<typename _CharT, typename _Traits>
 class basic_ostream {
 public:
 typedef basic_ostream<_Traits> __ostream_type;
 __ostream_type&
 operator<<(__ostream_type& (__ostream_type&))
 {
 }
 };
 template <typename _CharT, typename _Traits>
 basic_ostream<_Traits>&
 operator<<(basic_ostream<_CharT, _Traits>& __out, char* __s)
 {
 __ostream_insert(__out, __s,
 static_cast<streamsize>(_Traits::length(__s)));
 }
 template<typename _CharT, typename _Traits>
 basic_ostream<_CharT, _Traits>&
 endl(basic_ostream<_CharT, _Traits>& _os)
 { }
 ostream cout;
 ios_base::Init _it;
}
int main() {
 std::cout << "Hellorld!+" << std::endl;
}