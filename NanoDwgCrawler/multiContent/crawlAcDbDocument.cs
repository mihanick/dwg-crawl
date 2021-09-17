using DwgDump.Data;
using DwgDump.Enitites;
using Multicad;
using Multicad.DatabaseServices;
using Multicad.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace DwgDump
{
	class CrawlAcDbDocument
	{
		public readonly CrawlDocument CrawlDoc;

		public string FullPath => CrawlDoc.Path;
		public string FileId => CrawlDoc.FileId;

		public McDocument Document;

		private DbMongo Db => DbMongo.Instance;

		// https://stackoverflow.com/questions/2987559/check-if-a-file-is-open
		private bool IsFileinUse(FileInfo file)
		{
			FileStream stream = null;

			try
			{
				stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None);
			}
			catch (IOException)
			{
				//the file is unavailable because it is:
				//still being written to
				//or being processed by another thread
				//or does not exist (has already been processed)
				return true;
			}
			finally
			{
				if (stream != null)
					stream.Close();
			}

			return false;
		}

		public CrawlAcDbDocument(CrawlDocument crawlDoc)
		{
			this.CrawlDoc = crawlDoc;

			string path = Path.Combine(DbMongo.Instance.DataDir, crawlDoc.FileId + ".dwg");
		}

		public void DoOpen()
		{
			FileInfo info = new FileInfo(this.FullPath);
			if (!IsFileinUse(info))
			{
				this.Document = McDocumentsManager.OpenDocument(this.FullPath, true);

				DumpDocumentDescription();
			}
		}

		private void DumpDocumentDescription()
		{
			try
			{
				// also run all blocks
				// var ids = this.Document.GetBlocks();
				// TODO: Maybe we don't need it for now

				// all layer definitions
				var layersJsons = new List<string>();
				foreach (var layername in McObjectManager.CurrentStyle.GetLayers())
				{
					var layerRecord = McObjectManager.CurrentStyle.GetLayer(layername);

					string objId = layerRecord.ToString();

					LayerTableRecord cltr = new LayerTableRecord()
					{
						Name = layerRecord.Name,

						// BUG: Not implemented
						// this.Linetype = layerRecord.LineType.Name;

						LineWeight = layerRecord.LineWeight.ToString(),
						IsFrozen = layerRecord.IsFrozen,
						// BUG: Not implemented
						// this.IsHidden = layerRecord.IsHidden;
						IsOff = layerRecord.IsOff,
						IsPlottable = layerRecord.IsPlottable,
						Color = layerRecord.Color.ToString(),

						ObjectId = layerRecord.ID.ToString()
					};

					layersJsons.Add(Converters.Serialize(cltr));
				}

				this.Db.UpdateFileLayers(layersJsons, this.FileId);

				// Run all xrefs
				// List<McObjectId> xrefs = this.Document.GetXRefs();
				// TODO: Beacuse in current dataset no xref will be present
			}
			catch (Exception e)
			{
				CrawlDebug.WriteLine(e.Message);
				// Cannot dump layers and xrefs
			}
		}

		public void DumpEntireFile()
		{
			// If document wasn't loaded correctly
			if (this.Document == null)
				return;
			// nanoCAD can crash, or exception or whatever...
			// so there will be only one try per document
			// so we first set document scanned, 
			// than try to process it
			Db.SetDocumentScanned(this.FileId);

			var filter = new ObjectFilter();
			filter.AddDoc(this.Document);
			filter.AllObjects = true;

			// каждый набор объектов со своим Guid группы
			var groupId = Guid.NewGuid().ToString();
			DumpFragment(filter.GetObjects(), this.FileId, groupId);
		}

		public static void DumpFragment(List<McObjectId> entityIds, string fileId, string groupId)
		{
			var scanData = Converters.ConvertEntities2json(entityIds, fileId, groupId);
			DbMongo.Instance.SaveObjectData(scanData);
		}

		public static void DumpFragmentDescription(string fileId, string groupId, Bitmap annotated, BoundBlock bound, Bitmap stripped)
		{
			var folder = Path.Combine(DbMongo.Instance.DataDir, "images");
			var annotatatedFileName = Path.Combine(folder, "annotated_" + groupId + ".png");
			var strippedFileName = Path.Combine(folder, "stripped_" + groupId + ".png");

			annotated.Save(annotatatedFileName, ImageFormat.Png);
			stripped.Save(strippedFileName, ImageFormat.Png);

			var crawlFragment = new CrawlFragment()
			{
				FileId = fileId,
				GroupId = groupId,
				AnnotatedFileName = annotatatedFileName,
				StrippedFileName = strippedFileName,
				MaxBoundPoint = Converters.Pt(bound.MinPoint),
				MinBoundPoint = Converters.Pt(bound.MaxPoint)
			};

			DbMongo.Instance.SaveFragmentData(Converters.Serialize(crawlFragment));
		}
	}
}
