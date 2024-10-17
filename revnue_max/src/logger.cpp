#ifndef logger_cpp
#define logger_cpp
#include <fstream>
#include <string>
#include <ctime>
#include <iostream>
//#include "util.cpp"
using namespace std;

enum LogType { ERROR = -3, WARN = -2, OUTPUT = -1, INFO = 0, DEBUG = 1, TRACE = 2 };

class Logger {
public:
  LogType loglevel;
  ostream& of;
  bool echo;
  string msg;
  LogType msglevel;
  char buff[70];
   bool enabled;
  Logger() : of (cout) {
    //    of.open("log.txt");
    //    of = cout;
    loglevel = INFO;
    msglevel = INFO;
    echo = false;
    enabled = true;
  }
  
   Logger( LogType inlevel, ostream& os, bool echo_in = false ): of( os ) {
    //of.open("log.txt");
    //    of = cout;
      loglevel = inlevel;
      msglevel = INFO;
      echo = echo_in;
      enabled = true;
   }

   void set_level( LogType Lin ) {
      loglevel = Lin;
   }

   void operator()( LogType level, string msg ) {
      
      if (enabled) {
	 time_t t = time( NULL );
	 strftime( buff, sizeof( buff ), "%b %d %H:%M:%S", localtime( &t ) );
	 string sbuff( buff );
	 if (level <= loglevel) {
	    switch (level) {
	    case ERROR:
	       of << sbuff << "\033[31m [ERROR] \033[0m" << msg << endl;
	       break;
	    case WARN:
	       of << sbuff << "\033[33m [WARN] \033[0m" << msg << endl;
	       break;
	    case OUTPUT:
	       of << sbuff << " [OUTPUT] " << msg << endl;
	       break;
	    case INFO:
	       of << sbuff << "\033[32m [INFO] \033[0m" << msg << endl;
	       break;
	    case DEBUG:
	       of << sbuff << "\033[31m [DEBUG] \033[0m" << msg << endl;
	       break;
	    case TRACE:
	       of << sbuff << "\033[31m [TRACE] \033[0m" << msg << endl;
	       break;
	    }
	    if (echo) {
	       if (level <= loglevel) {
		  switch (level) {
		  case ERROR:
		     cout << sbuff << "\033[31m [ERROR] \033[0m" << msg << endl;
		     break;
		  case WARN:
		     cout << sbuff << "\033[33m [WARN] \033[0m" << msg << endl;
		     break;
		  case OUTPUT:
		     cout << sbuff << " [OUTPUT] " << msg << endl;
		     break;
		  case INFO:
		     cout << sbuff << "\033[32m [INFO] \033[0m" << msg << endl;
		     break;
		  case DEBUG:
		     cout << sbuff << "\033[31m [DEBUG] \033[0m" << msg << endl;
		     break;
		  case TRACE:
		     cout << sbuff << "\033[31m [TRACE] \033[0m" << msg << endl;
		     break;
		  }
	       }
	    }
	 }
      }
  }

  ~Logger() {
    //of.close();
  }
};

class endlclass {
  
} endL;

template <typename T>
Logger& operator<<( Logger& lhs, T word ) {
  lhs.msg += to_string(word);
  return lhs;
}

Logger& operator<<( Logger& lhs, const char* word ) {
  string tmp( word );
  lhs.msg += tmp;
  return lhs;
}

Logger& operator<<( Logger& lhs, string word ) {
  lhs.msg += word;
  return lhs;
}

Logger& operator<<( Logger& lhs, endlclass word ) {
  lhs( lhs.msglevel, lhs.msg );
  lhs.msg.clear();

  return lhs;
}

Logger& operator<<( Logger& lhs, LogType inmsgLevel ) {
  lhs.msglevel = inmsgLevel;

  return lhs;
}



#endif
