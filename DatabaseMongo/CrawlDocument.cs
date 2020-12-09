using System;
using System.Runtime.Serialization;
using System.IO;

namespace DwgDump.Data
{
    [DataContract]
    public class CrawlDocument
    {
        [DataMember]
        public string ClassName;
        [DataMember]
        public string Path;
        [DataMember]
        public string FileId;
        [DataMember]
        public string Hash;
        [DataMember]
        public bool Scanned;

        public CrawlDocument(string dwgPath = "")
        {

            this.Path = dwgPath;
            if (File.Exists(dwgPath))
            {
                this.Hash = DwgDump.Util.UtilityHash.HashFile(dwgPath);
                this.ClassName = "File";
            }

            this.FileId = Guid.NewGuid().ToString();
            this.Scanned = false;
        }
    }
}
