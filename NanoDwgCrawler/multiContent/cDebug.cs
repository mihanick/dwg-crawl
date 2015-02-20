using HostMgd.ApplicationServices;


namespace Crawl
{
    using System;
    using System.Diagnostics;
    using System.IO;

    /// <summary>
    /// Вывод сообщений в командную строку или DEBUG
    /// </summary>
    public class cDebug
    {
        public enum DebugMessageWriteMedia
        {
            CommandLine,
            Debug,
            LogFile
        }
        //DEBUG: для целей дебага по умолчанию сообщения будут выводиться в командную строку:

        /// <summary>
        /// Выводит строковое сообщение
        /// </summary>
        /// <param name="Message">Выводимое сообщение</param>
        /// <param name="Media">Куда выводить сообщение: CommandLine\Debug\LogFile</param>
        public static void WriteLine(string Message, DebugMessageWriteMedia Media = DebugMessageWriteMedia.Debug)
        {
            if (Media == DebugMessageWriteMedia.Debug)
            {
                Debug.WriteLine(Message);
            }
            if (Media == DebugMessageWriteMedia.CommandLine)
            {
                Application.DocumentManager.MdiActiveDocument.Editor.WriteMessage(Message);
            }
            if (Media == DebugMessageWriteMedia.LogFile)
            {
                try
                {
                    string logFileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "debug.log");
                    StreamWriter writer = new StreamWriter(logFileName);

                    writer.WriteLine(Message);

                    writer.Dispose();
                    writer.Close();
                }
                catch (System.Exception ex)
                {
                    cDebug.WriteLine(ex.Message);
                }
            }
        }

        /// <summary>
        /// Выводит форматированную строку - сообщение
        /// </summary>
        /// <param name="FormattedMessage">Форматированная строка</param>
        /// <param name="args">Аргументы форматированной строки</param>
        public static void WriteLine(string FormattedMessage, params Object[] args)
        {
            WriteLine(string.Format(FormattedMessage, args));
        }
    }
}
