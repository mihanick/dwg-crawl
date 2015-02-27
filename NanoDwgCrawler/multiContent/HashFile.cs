using System.Security.Cryptography;
using System.Text;
using System.IO;
using System.Diagnostics;
namespace Crawl
{
    //http://stackoverflow.com/questions/2150455/how-do-i-create-an-md5-hash-digest-from-a-text-file
    public static class UtilityHash
    {
        [DebuggerStepThrough]
        public static string HashFile(string filePath)
        {
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return HashFile(fs);
            }
        }

        public static string HashFile(FileStream stream)
        {
            StringBuilder sb = new StringBuilder();

            if (stream != null)
            {
                stream.Seek(0, SeekOrigin.Begin);

                MD5 md5 = MD5CryptoServiceProvider.Create();
                byte[] hash = md5.ComputeHash(stream);
                foreach (byte b in hash)
                    sb.Append(b.ToString("x2"));

                stream.Seek(0, SeekOrigin.Begin);
            }

            return sb.ToString();
        }
    }
}